
#!/usr/bin/env python3
"""Analyze ChampSim results and plot BTB metrics."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_FOLDER = "results_25M"
BASELINE_MODEL = "convBTB"


MODELS = [
    "convBTB",
    "convBTB_priority",
    "lfu",
    "lfu_priority",
    "lfu_priority_target",
    "lfu_priority_target_ind",
    "lfu_priority_target_reuse_d",
    "lfu_target_priority",
    "lru_target",
    "target_lfu_priority",
    "conditional_first",
    "call_first",
    "indirect_first",
    "return_first",
]

# Baseline policies: overall and per-trace plots.
BASELINE_MODELS = [
    "convBTB",
    "lfu",
    "lfu_priority",
    "lfu_priority_target_ind",
]

# Ordering experiments: overall plots only.
ORDERING_MODELS = [
    "lfu_target_priority",
    "lfu_priority_target",
    "lfu_priority_target_reuse_d",
    "lru_target",
    "target_lfu_priority",
]

# Branch ordering and priority experiments: overall plots only.
PRIORITY_MODELS = [
    "indirect_first",
    "call_first",
    "convBTB_priority",
    "return_first",
    "conditional_first",
]

# Names shown in plots. CSV files retain original model names.
DISPLAY_NAMES = {
    "convBTB": "LRU",
    "lfu": "LFU",
    "lfu_priority": "LFU + Priority",
    "lfu_priority_target": "LFU + Priority (Ind + Ret) + Target",
    "lfu_priority_target_ind": "LFU + Priority(Indirect) + Target",
    "lfu_priority_target_reuse_d": "LFU + Priority + Target + Reuse",
    "lfu_target_priority": "LFU + Target + Priority",
    "lru_target": "LRU + Target",
    "target_lfu_priority": "Target + LFU + Priority",
    "conditional_first": "Conditional Priority",
    "call_first": "Call Priority",
    "indirect_first": "Indirect Priority",
    "return_first": "Return Priority",
    "convBTB_priority": "Indirect + Return Priority",
}

PLOT_ORDER = [
    "convBTB",
    "lfu",
    "lfu_priority",
    "lfu_priority_target",
    "lfu_priority_target_ind",
    "lfu_priority_target_reuse_d",
    "lfu_target_priority",
    "lru_target",
    "target_lfu_priority",
    "indirect_first",
    "call_first",
    "convBTB_priority",
    "return_first",
    "conditional_first",
]

TRACES = [
    "spec_gcc_001",
    "spec_gobmk_001",
    "spec_perlbench_001",
    "spec_x264_001",
    "client_001",
    "client_005",
    "server_001",
    "server_010",
]

METRICS = {
    "ipc": ("IPC", "Higher is better"),
    "branch_accuracy_pct": (
        "Branch prediction accuracy (%)", "Higher is better"
    ),
    "branch_mpki": ("Branch MPKI", "Lower is better"),
    "l1d_miss_rate_pct": ("L1D miss rate (%)", "Lower is better"),
    "l2c_miss_rate_pct": ("L2C miss rate (%)", "Lower is better"),
    "llc_miss_rate_pct": ("LLC miss rate (%)", "Lower is better"),
    "branch_mispredictions": ("Branch mispredictions", "Lower is better"),
    "btb_reads": ("BTB reads", "Compare with IPC and MPKI"),
    "btb_writes": ("BTB writes", "Compare with IPC and MPKI"),
}

COLOURS = list(plt.get_cmap("tab20").colors)
HATCHES = [
    "", "///", "...", "\\\\\\", "xxx", "++", "oo",
    "**", "||", "--", "..", "//", "\\\\",
]

MODEL_STYLE = {
    model: (
        COLOURS[i % len(COLOURS)],
        HATCHES[i % len(HATCHES)],
    )
    for i, model in enumerate(PLOT_ORDER)
}


def num(value):
    value = value.replace(",", "").strip()
    return int(value) if re.fullmatch(r"\d+", value) else float(value)


def get_value(text, pattern):
    matches = re.findall(pattern, text, re.I | re.M)
    return num(matches[-1]) if matches else None


def identify(path, root):
    name = "_".join(path.relative_to(root).parts)

    model = next(
        (m for m in sorted(MODELS, key=len, reverse=True) if m in name),
        path.parent.name,
    )
    trace = next((t for t in TRACES if t in name), "unknown")

    return model, trace


def parse_result(path, root):
    text = path.read_text(errors="replace")
    model, trace = identify(path, root)

    row = {
        "model": model,
        "trace": trace,
        "result_file": str(path),
    }

    row["ipc"] = get_value(
        text, r"^\s*XXX\s+cumulative[-_]IPC\s+([0-9.eE+-]+)"
    )
    row["instructions"] = get_value(
        text, r"^\s*XXX\s+instructions\s+([0-9,]+)"
    )
    row["branch_mispredictions"] = get_value(
        text, r"^\s*XXX\s+total_mispredicts\s+([0-9,]+)"
    )

    bp = re.search(
        r"Branch Prediction Accuracy:\s*([0-9.]+)%.*?MPKI:\s*([0-9.eE+-]+)",
        text,
        re.I | re.S,
    )

    if bp:
        row["branch_accuracy_pct"] = float(bp.group(1))
        row["branch_mpki"] = float(bp.group(2))
    else:
        row["branch_accuracy_pct"] = get_value(
            text, r"Branch Prediction Accuracy:\s*([0-9.]+)%"
        )
        row["branch_mpki"] = get_value(
            text, r"\bMPKI:\s*([0-9.eE+-]+)"
        )

    for counter in ("btb_reads", "btb_writes"):
        row[counter] = get_value(
            text, rf"^\s*XXX\s+{counter}\s+([0-9,]+)"
        )

    for cache in ("L1D", "L2C", "LLC"):
        prefix = cache.lower()

        found = re.search(
            rf"\b{cache}\s+TOTAL\s+ACCESS:\s*([0-9,]+)\s+"
            rf"HIT:\s*([0-9,]+)\s+MISS:\s*([0-9,]+)",
            text,
            re.I,
        )

        if found:
            accesses, hits, misses = (
                int(value.replace(",", ""))
                for value in found.groups()
            )

            row[f"{prefix}_accesses"] = accesses
            row[f"{prefix}_hits"] = hits
            row[f"{prefix}_misses"] = misses

            if accesses:
                row[f"{prefix}_miss_rate_pct"] = 100 * misses / accesses

    return row


def fmt(value):
    if pd.isna(value):
        return ""

    if float(value).is_integer():
        return str(int(value))

    return format(float(value), ".15g")


def plot_metric(data, metric, title, output):
    data = data.dropna(subset=[metric]).copy()

    if data.empty:
        return False

    order = [m for m in PLOT_ORDER if m in data["model"].unique()]
    order += [m for m in data["model"].unique() if m not in order]

    data["order"] = data["model"].map(
        {model: i for i, model in enumerate(order)}
    )
    data = data.sort_values("order").reset_index(drop=True)

    values = data[metric].astype(float)
    low = values.min()
    high = values.max()
    span = high - low

    padding = span * 0.22 if span else max(abs(high) * 0.02, 0.01)
    bottom = max(0, low - padding) if low >= 0 else low - padding
    top = high + padding

    if top <= bottom:
        top = bottom + 1

    labels = [DISPLAY_NAMES.get(m, m) for m in data["model"]]

    fig, ax = plt.subplots(figsize=(max(11, len(data) * 0.95), 6))

    bars = ax.bar(
        range(len(data)),
        values,
        color=[
            MODEL_STYLE.get(model, ("grey", ""))[0]
            for model in data["model"]
        ],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, model in zip(bars, data["model"]):
        bar.set_hatch(MODEL_STYLE.get(model, ("grey", ""))[1])

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=50, ha="right")
    ax.set_ylim(bottom, top)
    ax.set_ylabel(METRICS[metric][0])
    ax.set_title(
        f"{title}\n{METRICS[metric][1]} | "
        f"y-axis: {bottom:.6g} to {top:.6g}"
    )

    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)

    offset = (top - bottom) * 0.015

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return True


# ============================================================
#       Geometric mean of normalized IPC speedups
# ============================================================

def calculate_geomean_speedup(df, baseline=BASELINE_MODEL):
    """
    Compute geometric mean of per-trace IPC speedups relative to convBTB.

    Per trace:
        speedup = IPC(policy, trace) / IPC(convBTB, trace)

    Across traces:
        GM speedup = exp(mean(log(speedup)))

    Only traces with positive IPC for both the policy and baseline
    are included. The baseline is taken from the full dataset, not
    from an individual experiment group.
    """
    if "ipc" not in df.columns:
        raise ValueError("IPC data is required to calculate speedups.")

    ipc = df[["model", "trace", "ipc"]].copy()
    ipc["ipc"] = pd.to_numeric(ipc["ipc"], errors="coerce")

    ipc = (
        ipc.groupby(["model", "trace"], as_index=False)["ipc"]
        .mean()
    )

    baseline_df = (
        ipc[ipc["model"] == baseline][["trace", "ipc"]]
        .rename(columns={"ipc": "baseline_ipc"})
    )

    if baseline_df.empty:
        raise ValueError(
            f"Baseline model '{baseline}' was not found. "
            "Check the model names parsed from your result filenames."
        )

    rows = []

    for model in ipc["model"].unique():
        policy_df = ipc[ipc["model"] == model][["trace", "ipc"]]

        paired = policy_df.merge(
            baseline_df,
            on="trace",
            how="inner",
        )

        paired = paired[
            paired["ipc"].notna()
            & paired["baseline_ipc"].notna()
            & np.isfinite(paired["ipc"])
            & np.isfinite(paired["baseline_ipc"])
            & (paired["ipc"] > 0)
            & (paired["baseline_ipc"] > 0)
        ].copy()

        if paired.empty:
            geomean_speedup = np.nan
        else:
            speedups = (
                paired["ipc"].to_numpy(dtype=float)
                / paired["baseline_ipc"].to_numpy(dtype=float)
            )

            geomean_speedup = float(
                np.exp(np.log(speedups).mean())
            )

        rows.append({
            "model": model,
            "ipc_geomean_speedup": geomean_speedup,
            "speedup_trace_count": len(paired),
        })

    return pd.DataFrame(rows)


def plot_geomean_speedup(data, title, output):
    data = data.dropna(subset=["ipc_geomean_speedup"]).copy()

    if data.empty:
        return False

    order = [m for m in PLOT_ORDER if m in data["model"].unique()]
    order += [m for m in data["model"].unique() if m not in order]

    data["order"] = data["model"].map(
        {model: i for i, model in enumerate(order)}
    )
    data = data.sort_values("order").reset_index(drop=True)

    values = data["ipc_geomean_speedup"].astype(float)
    labels = [DISPLAY_NAMES.get(m, m) for m in data["model"]]

    fig, ax = plt.subplots(figsize=(max(11, len(data) * 0.95), 6))

    bars = ax.bar(
        range(len(data)),
        values,
        color=[
            MODEL_STYLE.get(model, ("grey", ""))[0]
            for model in data["model"]
        ],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, model in zip(bars, data["model"]):
        bar.set_hatch(MODEL_STYLE.get(model, ("grey", ""))[1])

    # Baseline reference: a speedup of exactly 1.0.
    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="LRU baseline = 1.0",
    )

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=50, ha="right")
    ax.set_ylabel("Geometric mean IPC speedup vs LRU")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend()

    low = min(float(values.min()), 1.0)
    high = max(float(values.max()), 1.0)
    span = high - low
    padding = span * 0.15 if span else 0.02

    bottom = max(0, low - padding)
    top = high + padding
    ax.set_ylim(bottom, top)

    offset = (top - bottom) * 0.015

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return True


def make_plots(df, metrics, plots_dir, speedup_summary):
    plots_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        "main": BASELINE_MODELS + [
            model for model in ORDERING_MODELS
            if model not in BASELINE_MODELS
        ],
        "ordering_experiments": ORDERING_MODELS,
        "priority_experiments": PRIORITY_MODELS,
    }

    count = 0

    for group_name, models in groups.items():
        group_df = df[df["model"].isin(models)]

        group_dir = (
            plots_dir
            if group_name == "main"
            else plots_dir / group_name
        )
        group_dir.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            overall = group_df.groupby(
                "model", as_index=False
            )[metric].mean()

            count += plot_metric(
                overall,
                metric,
                f"{METRICS[metric][0]} by BTB policy, all traces",
                group_dir / f"{metric}_overall.png",
            )

        if group_name == "main":
            trace_dir = group_dir / "per_trace"
            trace_dir.mkdir(parents=True, exist_ok=True)

            for trace in sorted(group_df["trace"].dropna().unique()):
                if trace == "unknown":
                    continue

                trace_df = group_df[group_df["trace"] == trace]

                for metric in metrics:
                    summary = trace_df.groupby(
                        "model", as_index=False
                    )[metric].mean()

                    count += plot_metric(
                        summary,
                        metric,
                        f"{METRICS[metric][0]} by BTB policy, trace {trace}",
                        trace_dir / f"{metric}_{trace}.png",
                    )

        # Normalized geometric mean IPC speedup plot per group.
        # The speedups were computed using the full dataset, so the
        # baseline need not be part of the current experiment group.
        group_speedup = speedup_summary[
            speedup_summary["model"].isin(models)
        ]

        count += plot_geomean_speedup(
            group_speedup,
            f"Geometric Mean IPC Speedup vs LRU — {group_name}",
            group_dir / "ipc_geomean_speedup.png",
        )

    # NEW: one plot comparing all available policies.
    count += plot_geomean_speedup(
        speedup_summary,
        "Geometric Mean IPC Speedup vs LRU (convBTB)",
        plots_dir / "ipc_geomean_speedup.png",
    )

    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(RESULTS_FOLDER),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out = (args.out or root / "analysis").expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Result directory not found: {root}")

    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if out == path or out in path.parents or ".git" in path.parts:
            continue

        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue

        if (
            "XXX cumulative-IPC" in text
            or "Region of Interest Statistics" in text
        ):
            files.append(path)

    if not files:
        raise SystemExit(f"No ChampSim result files found under {root}")

    rows = []

    for path in sorted(files):
        try:
            rows.append(parse_result(path, root))
        except (OSError, ValueError) as exc:
            print(f"Skipping {path}: {exc}")

    df = pd.DataFrame(rows)

    if df.empty:
        raise SystemExit("No result files could be parsed.")

    out.mkdir(parents=True, exist_ok=True)

    # Original per-run CSV.
    df.to_csv(
        out / "metrics_per_run.csv",
        index=False,
        float_format="%.15g",
    )

    present = [
        metric
        for metric in METRICS
        if metric in df and df[metric].notna().any()
    ]

    if not present:
        raise SystemExit("None of the requested metrics were parsed.")

    summary = df.groupby("model", as_index=False)[present].mean()

    # Calculate normalized geometric-mean speedups once from
    # the complete dataset, before filtering for individual groups.
    if "ipc" not in df.columns or not df["ipc"].notna().any():
        raise SystemExit("No IPC values were parsed; cannot calculate speedup.")

    try:
        speedup_summary = calculate_geomean_speedup(df)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Add normalized GM results and coverage to the existing summary.
    summary = summary.merge(
        speedup_summary,
        on="model",
        how="left",
    )

    summary.to_csv(
        out / "comparison_metrics_by_model.csv",
        index=False,
        float_format="%.15g",
    )

    df[["model", "trace"] + present].to_csv(
        out / "comparison_metrics_per_trace.csv",
        index=False,
        float_format="%.15g",
    )

    speedup_summary.to_csv(
        out / "ipc_geomean_speedup.csv",
        index=False,
        float_format="%.15g",
    )

    plots_dir = out / "plots"
    count = make_plots(
        df,
        present,
        plots_dir,
        speedup_summary,
    )

    pd.DataFrame({
        "metric": present,
        "runs_with_value": [
            int(df[metric].notna().sum()) for metric in present
        ],
        "runs_missing_value": [
            int(df[metric].isna().sum()) for metric in present
        ],
    }).to_csv(out / "metric_coverage.csv", index=False)

    print(f"Read {len(rows)} result files.")
    print(f"Created {count} plots.")
    print(f"Main overall plots: {plots_dir}")
    print(f"Baseline per-trace plots: {plots_dir / 'per_trace'}")
    print(
        "Ordering experiment plots: "
        f"{plots_dir / 'ordering_experiments'}"
    )
    print(
        "Priority experiment plots: "
        f"{plots_dir / 'priority_experiments'}"
    )
    print(f"CSV files saved in {out}")

    print("\nGeometric mean IPC speedup vs LRU:")
    print(
        speedup_summary.sort_values(
            "ipc_geomean_speedup",
            ascending=False,
            na_position="last",
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
