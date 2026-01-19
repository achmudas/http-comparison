#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="analysis_out/summary.csv", help="Path to summary.csv")
    ap.add_argument("--outdir", default="analysis_out/charts", help="Charts output dir")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    s = pd.read_csv(args.summary)

    # ---- Chart 1: Protocol vs total load time (per scenario, baseline=good) ----
    for scenario in sorted(s["scenario"].dropna().unique()):
        d = s[(s["scenario"] == scenario) & (s["profile"] == "good")].copy()
        if d.empty:
            continue

        # bar chart: protocol -> median_total_s
        d = d.sort_values("http_version")
        plt.figure()
        plt.bar(d["http_version"], d["median_total_s"])
        plt.ylabel("Median total time (s)")
        plt.xlabel("HTTP version")
        plt.title(f"{scenario}: Protocol vs median total time (good network)")
        savefig(os.path.join(args.outdir, f"protocol_vs_total_good__{scenario}.png"))

    # ---- Chart 2: Packet loss vs performance (compare profiles for each protocol) ----
    # We'll plot median total time across profiles for each scenario+protocol.
    profiles_order = ["good", "mobile", "lossy"]

    for scenario in sorted(s["scenario"].dropna().unique()):
        for proto in sorted(s["http_version"].dropna().unique()):
            d = s[(s["scenario"] == scenario) & (s["http_version"] == proto)].copy()
            if d.empty:
                continue

            # enforce profile order
            d["profile"] = pd.Categorical(d["profile"], categories=profiles_order, ordered=True)
            d = d.sort_values("profile")

            plt.figure()
            plt.plot(d["profile"], d["median_total_s"], marker="o")
            plt.ylabel("Median total time (s)")
            plt.xlabel("Network profile")
            plt.title(f"{scenario}: profile vs median total ({proto})")
            savefig(os.path.join(args.outdir, f"profile_vs_total__{scenario}__{proto}.png"))

    # ---- Chart 3: Scenario comparison (per profile, show protocols side-by-side) ----
    for profile in sorted(s["profile"].dropna().unique()):
        d = s[s["profile"] == profile].copy()
        if d.empty:
            continue

        # pivot: rows=scenario, cols=protocol, values=median_total_s
        pivot = d.pivot_table(
            index="scenario",
            columns="http_version",
            values="median_total_s",
            aggfunc="mean",
        )

        plt.figure(figsize=(10, 5))
        pivot.plot(kind="bar")
        plt.ylabel("Median total time (s)")
        plt.xlabel("Scenario")
        plt.title(f"Scenario comparison (profile={profile})")
        plt.legend(title="HTTP version")
        savefig(os.path.join(args.outdir, f"scenario_comparison__{profile}.png"))

    # ---- Optional: p95 charts (useful for latency tails) ----
    for scenario in sorted(s["scenario"].dropna().unique()):
        d = s[(s["scenario"] == scenario) & (s["profile"] == "lossy")].copy()
        if d.empty:
            continue

        d = d.sort_values("http_version")
        plt.figure()
        plt.bar(d["http_version"], d["p95_total_s"])
        plt.ylabel("p95 total time (s)")
        plt.xlabel("HTTP version")
        plt.title(f"{scenario}: p95 total time (lossy network)")
        savefig(os.path.join(args.outdir, f"p95_total_lossy__{scenario}.png"))

    print(f"Wrote charts to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
