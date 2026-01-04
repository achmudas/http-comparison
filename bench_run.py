#!/usr/bin/env python3
"""
bench_run.py — Run HTTP benchmarks using curl and save JSONL results.

Supports:
- YAML or JSON scenario files
- Protocol forcing: h1, h2, h3
- Metrics via curl --write-out: total time, TTFB, bytes
- Repetitions
- Concurrency (parallel requests) using asyncio + Semaphore
- JSONL output: one record per request
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------- Scenario loading (YAML or JSON) ----------

def load_scenario(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            raise SystemExit(
                "PyYAML is required for YAML scenarios.\n"
                "Install: python3 -m pip install pyyaml\n"
                "Or use a .json scenario file."
            )
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    else:
        raise SystemExit("Scenario must be .yaml/.yml or .json")


# ---------- URL expansion helpers ----------

def _parse_brace_range(token: str) -> Optional[Tuple[str, str]]:
    # token like "{001..100}" or "{1..10}"
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
    start_idx = s.find("{")
    end_idx = s.find("}", start_idx + 1)
    if start_idx == -1 or end_idx == -1:
        return [s]

    token = s[start_idx:end_idx + 1]
    parsed = _parse_brace_range(token)
    if not parsed:
        return [s]

    a, b = parsed
    if not (a.isdigit() and b.isdigit()):
        return [s]

    ia, ib = int(a), int(b)
    step = 1 if ib >= ia else -1

    # Preserve zero-padding if start/end have same width
    pad = len(a) if len(a) == len(b) else 0

    out: List[str] = []
    for i in range(ia, ib + step, step):
        num = str(i).zfill(pad) if pad else str(i)
        out.append(s[:start_idx] + num + s[end_idx + 1:])
    return out

def scenario_urls(scn: Dict[str, Any]) -> List[str]:
    """
    Supports:
      - url: "/single"
      - urls: ["/a", "/b"]
      - requests: [{url: "/x"}] or [{pattern: "/x/{001..100}"}]
      - entrypoint + assets (real page scenario)
    """
    urls: List[str] = []

    # Single URL
    if isinstance(scn.get("url"), str):
        urls.append(scn["url"])

    # List of URLs
    if isinstance(scn.get("urls"), list):
        urls.extend([u for u in scn["urls"] if isinstance(u, str)])

    # Request definitions (with optional brace expansion)
    if isinstance(scn.get("requests"), list):
        for item in scn["requests"]:
            if isinstance(item, dict):
                if isinstance(item.get("url"), str):
                    urls.append(item["url"])
                elif isinstance(item.get("pattern"), str):
                    urls.extend(expand_brace_ranges(item["pattern"]))

    # Real page scenario: entrypoint + assets
    if isinstance(scn.get("entrypoint"), str):
        urls.append(scn["entrypoint"])

        if isinstance(scn.get("assets"), list):
            urls.extend([a for a in scn["assets"] if isinstance(a, str)])

    # Expand brace ranges everywhere (idempotent if none exist)
    expanded: List[str] = []
    for u in urls:
        expanded.extend(expand_brace_ranges(u))

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for u in expanded:
        if u not in seen:
            seen.add(u)
            out.append(u)

    return out



# ---------- Curl execution ----------

PROTO_FLAGS = {
    "h1": ["--http1.1"],
    "h2": ["--http2"],
    "h3": ["--http3"],
}

@dataclass
class CurlResult:
    ok: bool
    http_code: Optional[int]
    http_version: Optional[str]
    time_total: Optional[float]
    ttfb: Optional[float]
    bytes_download: Optional[int]
    error: Optional[str]
    stderr: str

def build_curl_cmd(
    curl_bin: str,
    url: str,
    proto: str,
    insecure: bool,
) -> List[str]:
    write_out = (
        r'{"http_code":%{http_code},"http_version":"%{http_version}",'
        r'"time_total":%{time_total},"ttfb":%{time_starttransfer},'
        r'"bytes":%{size_download}}'
    )

    cmd = [curl_bin, "-sS", "-o", "/dev/null"]
    if insecure:
        cmd.append("-k")
    cmd += PROTO_FLAGS[proto]
    cmd += ["-w", write_out, url]
    return cmd

async def run_curl_async(
    sem: asyncio.Semaphore,
    curl_bin: str,
    url: str,
    proto: str,
    insecure: bool,
    timeout_s: int,
) -> CurlResult:
    if proto not in PROTO_FLAGS:
        return CurlResult(False, None, None, None, None, None, f"Unknown proto: {proto}", "")

    cmd = build_curl_cmd(curl_bin=curl_bin, url=url, proto=proto, insecure=insecure)

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
                    http_code=int(data.get("http_code")) if data.get("http_code") is not None else None,
                    http_version=str(data.get("http_version")) if data.get("http_version") is not None else None,
                    time_total=float(data.get("time_total")) if data.get("time_total") is not None else None,
                    ttfb=float(data.get("ttfb")) if data.get("ttfb") is not None else None,
                    bytes_download=int(data.get("bytes")) if data.get("bytes") is not None else None,
                    error=None,
                    stderr=stderr,
                )
            except Exception as e:
                return CurlResult(False, None, None, None, None, None, f"parse_error:{e}", stderr)
        except FileNotFoundError:
            return CurlResult(False, None, None, None, None, None, "curl_not_found", "")
        except Exception as e:
            return CurlResult(False, None, None, None, None, None, f"runner_error:{e}", "")


# ---------- Main runner ----------

async def run_benchmark(scn: Dict[str, Any], args: argparse.Namespace) -> None:
    name = str(scn.get("name", Path(args.scenario).stem))
    base_url = str(scn.get("base_url", "")).rstrip("/")

    urls = scenario_urls(scn)
    if not urls:
        raise SystemExit("No URLs found. Provide 'url', 'urls', or 'requests'.")

    # Join base_url + paths
    full_urls: List[str] = []
    for u in urls:
        if u.startswith("http://") or u.startswith("https://"):
            full_urls.append(u)
        else:
            full_urls.append(base_url + u)

    repetitions = int(args.reps) if args.reps > 0 else int(scn.get("repetitions", 1))

    # Concurrency (scenario default, CLI override)
    scn_conc = int(scn.get("concurrency", 1))
    concurrency = int(args.concurrency) if args.concurrency > 0 else scn_conc
    if concurrency < 1:
        concurrency = 1

    # Protocols
    if args.protocols.strip():
        protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    else:
        protocols = scn.get("protocols", ["h1", "h2", "h3"])
    protocols = [p for p in protocols if p in PROTO_FLAGS]
    if not protocols:
        raise SystemExit("No valid protocols. Use h1,h2,h3")

    insecure = bool(scn.get("insecure", False) or args.insecure)
    curl_bin = args.curl_bin
    timeout_s = int(args.timeout)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "type": "meta",
        "ts_unix": time.time(),
        "scenario": name,
        "base_url": base_url,
        "repetitions": repetitions,
        "protocols": protocols,
        "url_count": len(full_urls),
        "concurrency": concurrency,
        "curl_bin": curl_bin,
        "insecure": insecure,
        "note": "one JSON object per line; metrics from curl --write-out",
    }

    sem = asyncio.Semaphore(concurrency)

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        run_id = 0

        for rep in range(1, repetitions + 1):
            for proto in protocols:
                batch_id = f"{name}|rep={rep}|proto={proto}"

                # If concurrency == 1, run sequentially (simpler, easier to debug)
                if concurrency == 1:
                    for url in full_urls:
                        run_id += 1
                        res = await run_curl_async(sem, curl_bin, url, proto, insecure, timeout_s)
                        record = {
                            "type": "result",
                            "ts_unix": time.time(),
                            "scenario": name,
                            "rep": rep,
                            "batch_id": batch_id,
                            "run_id": run_id,
                            "protocol": proto,
                            "url": url,
                            "ok": res.ok,
                            "http_code": res.http_code,
                            "http_version": res.http_version,
                            "time_total_s": res.time_total,
                            "ttfb_s": res.ttfb,
                            "bytes": res.bytes_download,
                            "error": res.error,
                            "stderr": res.stderr,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                # Concurrency > 1: schedule tasks and gather as they finish
                tasks = []
                for url in full_urls:
                    run_id += 1
                    task = asyncio.create_task(
                        run_curl_async(sem, curl_bin, url, proto, insecure, timeout_s)
                    )
                    tasks.append((run_id, url, task))

                for rid, url, task in tasks:
                    res = await task
                    record = {
                        "type": "result",
                        "ts_unix": time.time(),
                        "scenario": name,
                        "rep": rep,
                        "batch_id": batch_id,
                        "run_id": rid,
                        "protocol": proto,
                        "url": url,
                        "ok": res.ok,
                        "http_code": res.http_code,
                        "http_version": res.http_version,
                        "time_total_s": res.time_total,
                        "ttfb_s": res.ttfb,
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
    ap.add_argument("--curl", dest="curl_bin", type=str, default="curl", help="curl binary path (e.g. $(brew --prefix curl)/bin/curl)")
    ap.add_argument("--protocols", type=str, default="", help="Comma list: h1,h2,h3 (overrides scenario)")
    ap.add_argument("--reps", type=int, default=0, help="Override repetitions")
    ap.add_argument("--concurrency", type=int, default=0, help="Override concurrency (parallel requests)")
    ap.add_argument("--insecure", action="store_true", help="Pass -k to curl (useful for https://localhost)")
    ap.add_argument("--timeout", type=int, default=60, help="curl timeout seconds")
    args = ap.parse_args()

    scn = load_scenario(Path(args.scenario))
    asyncio.run(run_benchmark(scn, args))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
