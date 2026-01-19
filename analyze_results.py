#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def infer_profile_from_filename(path: str) -> str:
    base = os.path.basename(path)
    for p in ("good", "mobile", "lossy"):
        if base.startswith(p + "_"):
            return p
    return "unknown"


def load_jsonl_results(files: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for fp in files:
        profile = infer_profile_from_filename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj.get("type") != "result":
                    continue

                # Keep even failed ones for debugging, but mark ok
                rows.append(
                    {
                        "file": fp,
                        "profile": profile,
                        "scenario": obj.get("scenario"),
                        "rep": obj.get("rep"),
                        "batch_id": obj.get("batch_id"),
                        "run_id": obj.get("run_id"),
                        "http_version_requested": obj.get("http_version_requested"),
                        "http_version_used": obj.get("http_version_used"),
                        "url": obj.get("url"),
                        "ok": bool(obj.get("ok")),
                        "http_code": obj.get("http_code"),
                        "time_total_s": obj.get("time_total_s"),
                        "ttfb_s": obj.get("ttfb_s"),
                        "bytes": obj.get("bytes"),
                        "error": obj.get("error"),
                    }
                )

    df = pd.DataFrame(rows)

    # Normalize types
    for col in ("time_total_s", "ttfb_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # Only successful runs with real timings
    d = df[(df["ok"] == True) & df["time_total_s"].notna() & df["ttfb_s"].notna()].copy()

    # p95 function
    def p95(x: pd.Series) -> float:
        return float(np.percentile(x.to_numpy(), 95))

    # group by profile+scenario+protocol
    summary = (
        d.groupby(["profile", "scenario", "http_version_requested"], dropna=False)
        .agg(
            n=("time_total_s", "count"),
            median_total_s=("time_total_s", "median"),
            p95_total_s=("time_total_s", p95),
            median_ttfb_s=("ttfb_s", "median"),
            p95_ttfb_s=("ttfb_s", p95),
            median_bytes=("bytes", "median"),
        )
        .reset_index()
        .rename(columns={"http_version_requested": "http_version"})
    )

    return summary.sort_values(["profile", "scenario", "http_version"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/*.jsonl", help="Input JSONL glob pattern")
    ap.add_argument("--outdir", default="analysis_out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    df = load_jsonl_results(files)
    df.to_csv(os.path.join(args.outdir, "all_results.csv"), index=False)

    summary = summarize(df)
    summary.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    print(f"Wrote: {args.outdir}/all_results.csv")
    print(f"Wrote: {args.outdir}/summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
