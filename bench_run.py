#!/usr/bin/env python3
"""
bench_run.py — HTTP/1.1 vs HTTP/2 vs HTTP/3 benchmark runner using curl.

What it does:
- Reads a scenario file (YAML or JSON)
- Builds a list of URLs from:
    - url / urls / requests(pattern|url)
    - entrypoint + assets   (real-page scenario)
- Runs each URL multiple times for each HTTP version, optionally concurrently
- Forces HTTP version via curl flags: --http1.1 / --http2 / --http3
- Captures metrics from curl --write-out:
    - time_total (seconds)
    - time_starttransfer (TTFB, seconds)
    - size_download (bytes)
- Writes JSONL:
    - one "meta" record at the start
    - one "result" record per request

Notes:
- `https` vs `http` is controlled by your URLs (base_url). This script compares HTTP *versions*.
- Use `insecure: true` in the scenario for https://localhost if you don't trust the local CA yet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------- Scenario loading (YAML or JSON) ----------------

def load_scenario(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            raise SystemExit(
                "PyYAML is required for YAML scenarios.\n"
                "Install inside a venv: python -m pip install pyyaml\n"
                "Or use a .json scenario file."
            )
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit("Scenario must be .yaml/.yml or .json")


# ---------------- URL expansion helpers ----------------

def _parse_brace_range(token: str) -> Optional[Tuple[str, str]]:
    # token like "{001..100}"
    if token.startswith("{") and token.endswith("}") and ".." in token:
        inner = token[1:-1]
        start, end = inner.split("..", 1)
        return start, end
    return None

def expand_brace_ranges(s: str) -> List[str]:
    """
    Expands one brace range occurrence like:
      "/x/{001..003}" -> ["/x/001", "/x/002", "/x/003"]

    If no brace range exists, returns [s].
    """
    l = s.find("{")
    r = s.find("}", l + 1)
    if l == -1 or r == -1:
        return [s]

    token = s[l:r + 1]
    parsed = _parse_brace_range(token)
    if not parsed:
        return [s]

    a, b = parsed
    if not (a.isdigit() and b.isdigit()):
        return [s]

    ia, ib = int(a), int(b)
    step = 1 if ib >= ia else -1
    pad = len(a) if len(a) == len(b) else 0

    out: List[str] = []
    for i in range(ia, ib + step, step):
        num = str(i).zfill(pad) if pad else str(i)
        out.append(s[:l] + num + s[r + 1:])
    return out

def scenario_urls(scn: Dict[str, Any]) -> List[str]:
    """
    Builds URL paths from the scenario.

    Supports:
      - url: "/single"
      - urls: ["/a", "/b"]
      - requests: [{url: "/x"}] or [{pattern: "/x/{001..100}"}]
      - entrypoint: "/page/" and assets: [...]
    """
    urls: List[str] = []

    # Single URL
    if isinstance(scn.get("url"), str):
        urls.append(scn["url"])

    # List of URLs
    if isinstance(scn.get("urls"), list):
        urls.extend([u for u in scn["urls"] if isinstance(u, str)])

    # Request patterns
    if isinstance(scn.get("requests"), list):
        for item in scn["requests"]:
            if isinstance(item, dict):
                if isinstance(item.get("url"), str):
                    urls.append(item["url"])
                elif isinstance(item.get("pattern"), str):
                    urls.extend(expand_brace_ranges(item["pattern"]))

    # Real page scenario
    if isinstance(scn.get("entrypoint"), str):
        urls.append(scn["entrypoint"])
        if isinstance(scn.get("assets"), list):
            urls.extend([a for a in scn["assets"] if isinstance(a, str)])

    # Expand brace ranges everywhere + de-dupe preserving order
    expanded: List[str] = []
    for u in urls:
        expanded.extend(expand_brace_ranges(u))

    seen = set()
    out: List[str] = []
    for u in expanded:
        if u not in seen:
            seen.add(u)
            out.append(u)

    return out


# ---------------- Curl execution ----------------

HTTP_VERSION_FLAGS = {
    "http1.1": ["--http1.1"],
    "http2": ["--http2"],
    "http3": ["--http3"],
}

@dataclass
class CurlResult:
    ok: bool
    http_code: Optional[int]
    http_version: Optional[str]
    time_total_s: Optional[float]
    ttfb_s: Optional[float]
    bytes_download: Optional[int]
    error: Optional[str]
    stderr: str

def build_curl_cmd(curl_bin: str, url: str, http_version_key: str, insecure: bool) -> List[str]:
    # JSON output makes parsing safe/robust.
    write_out = (
        r'{"http_code":%{http_code},"http_version":"%{http_version}",'
        r'"time_total":%{time_total},"ttfb":%{time_starttransfer},'
        r'"bytes":%{size_download}}'
    )
    cmd = [curl_bin, "-sS", "-o", "/dev/null"]
    if insecure:
        cmd.append("-k")
    cmd += HTTP_VERSION_FLAGS[http_version_key]
    cmd += ["-w", write_out, url]
    return cmd

async def run_curl_async(
    sem: asyncio.Semaphore,
    curl_bin: str,
    url: str,
    http_version_key: str,
    insecure: bool,
    timeout_s: int,
) -> CurlResult:
    if http_version_key not in HTTP_VERSION_FLAGS:
        return CurlResult(False, None, None, None, None, None, f"unknown_http_version:{http_version_key}", "")

    cmd = build_curl_cmd(curl_bin, url, http_version_key, insecure)

    async with sem:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                return CurlResult(False, None, None, None, None, None, "timeout", "")

            stderr = stderr_b.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                return CurlResult(False, None, None, None, None, None, f"curl_exit_{proc.returncode}", stderr)

            raw = stdout_b.decode("utf-8", errors="replace").strip()
            try:
                data = json.loads(raw)
                return CurlResult(
                    ok=True,
                    http_code=int(data["http_code"]) if data.get("http_code") is not None else None,
                    http_version=str(data["http_version"]) if data.get("http_version") is not None else None,
                    time_total_s=float(data["time_total"]) if data.get("time_total") is not None else None,
                    ttfb_s=float(data["ttfb"]) if data.get("ttfb") is not None else None,
                    bytes_download=int(data["bytes"]) if data.get("bytes") is not None else None,
                    error=None,
                    stderr=stderr,
                )
            except Exception as e:
                return CurlResult(False, None, None, None, None, None, f"parse_error:{e}", stderr)

        except FileNotFoundError:
            return CurlResult(False, None, None, None, None, None, "curl_not_found", "")
        except Exception as e:
            return CurlResult(False, None, None, None, None, None, f"runner_error:{e}", "")


# ---------------- Runner ----------------

def normalize_http_versions(scn: Dict[str, Any]) -> List[str]:
    """
    Preferred key: http_versions: ["http1.1","http2","http3"]
    Back-compat: protocols: ["h1","h2","h3"]  (mapped to http1.1/http2/http3)
    """
    if isinstance(scn.get("http_versions"), list):
        raw = [x for x in scn["http_versions"] if isinstance(x, str)]
    elif isinstance(scn.get("protocols"), list):
        # backward compatible: h1/h2/h3
        mapping = {"h1": "http1.1", "h2": "http2", "h3": "http3"}
        raw = []
        for x in scn["protocols"]:
            if isinstance(x, str):
                raw.append(mapping.get(x.strip(), x.strip()))
    else:
        raw = ["http1.1", "http2", "http3"]

    # filter only supported
    out = []
    for v in raw:
        v = v.strip()
        if v in HTTP_VERSION_FLAGS:
            out.append(v)
    return out

async def run_benchmark(scn: Dict[str, Any], args: argparse.Namespace) -> None:
    name = str(scn.get("name", Path(args.scenario).stem))
    base_url = str(scn.get("base_url", "")).rstrip("/")

    url_paths = scenario_urls(scn)
    if not url_paths:
        raise SystemExit("No URLs found. Provide url / urls / requests / or entrypoint+assets.")

    # Convert into full URLs
    full_urls: List[str] = []
    for u in url_paths:
        if u.startswith("http://") or u.startswith("https://"):
            full_urls.append(u)
        else:
            full_urls.append(base_url + u)

    repetitions = int(scn.get("repetitions", 1))
    concurrency = int(scn.get("concurrency", 1))
    if concurrency < 1:
        concurrency = 1

    http_versions = normalize_http_versions(scn)
    if not http_versions:
        raise SystemExit("No valid http_versions. Allowed: http1.1, http2, http3")

    insecure = bool(scn.get("insecure", False))
    curl_bin = str(scn.get("curl_bin", args.curl_bin))  # allow scenario override if desired
    timeout_s = int(scn.get("timeout_s", args.timeout))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)

    meta = {
        "type": "meta",
        "ts_unix": time.time(),
        "scenario": name,
        "base_url": base_url,
        "repetitions": repetitions,
        "http_versions": http_versions,
        "url_count": len(full_urls),
        "concurrency": concurrency,
        "curl_bin": curl_bin,
        "insecure": insecure,
        "timeout_s": timeout_s,
        "note": "one JSON object per line; timings in seconds from curl --write-out",
    }

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        run_id = 0

        for rep in range(1, repetitions + 1):
            for hv in http_versions:
                batch_id = f"{name}|rep={rep}|http={hv}"

                # schedule requests; semaphore enforces max parallelism
                tasks: List[Tuple[int, str, asyncio.Task[CurlResult]]] = []
                for url in full_urls:
                    run_id += 1
                    t = asyncio.create_task(
                        run_curl_async(sem, curl_bin, url, hv, insecure, timeout_s)
                    )
                    tasks.append((run_id, url, t))

                # write results in a stable order (by run_id)
                for rid, url, task in tasks:
                    res = await task
                    record = {
                        "type": "result",
                        "ts_unix": time.time(),
                        "scenario": name,
                        "rep": rep,
                        "batch_id": batch_id,
                        "run_id": rid,
                        "http_version_requested": hv,
                        "http_version_used": res.http_version,  # what curl reports
                        "url": url,
                        "ok": res.ok,
                        "http_code": res.http_code,
                        "time_total_s": res.time_total_s,
                        "ttfb_s": res.ttfb_s,
                        "bytes": res.bytes_download,
                        "error": res.error,
                        "stderr": res.stderr,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote results to: {out_path}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=str, help="Path to scenario .yaml/.yml or .json")
    ap.add_argument("--out", type=str, default="results.jsonl", help="Output JSONL file")
    ap.add_argument(
        "--curl",
        dest="curl_bin",
        type=str,
        default="curl",
        help='curl binary path (e.g. "$(brew --prefix curl)/bin/curl")',
    )
    ap.add_argument("--timeout", type=int, default=60, help="curl timeout seconds")
    args = ap.parse_args()

    scn = load_scenario(Path(args.scenario))
    asyncio.run(run_benchmark(scn, args))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
