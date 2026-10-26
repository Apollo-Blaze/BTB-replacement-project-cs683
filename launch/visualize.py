
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


RESULTS_FOLDER = "results_50M"
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

BASELINE_MODELS = [
    "convBTB",
    "lfu",
    "lfu_priority",
    "lfu_priority_target_ind",
]

ORDERING_MODELS = [
    "lfu_target_priority",
    "lfu_priority_target",
    "lfu_priority_target_reuse_d",
    "lru_target",
    "target_lfu_priority",
]

PRIORITY_MODELS = [
    "indirect_first",
    "call_first",
    "convBTB_priority",
    "return_first",
    "conditional_first",
]

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
    IPC ratio:
        policy IPC / baseline IPC

    GM speedup:
        exp(mean(log(per-trace IPC ratios)))

    Only traces with positive, finite IPC for both policies
    are included.
    """
    if "ipc" not in df.columns:
        raise ValueError("IPC data is required to calculate speedups.")

    ipc = df[["model", "trace", "ipc"]].copy()
    ipc["ipc"] = pd.to_numeric(ipc["ipc"], errors="coerce")

    ipc = ipc.groupby(
        ["model", "trace"], as_index=False
    )["ipc"].mean()

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
            baseline_df, on="trace", how="inner"
        )

        paired = paired[
            paired["ipc"].notna()
            & paired["baseline_ipc"].notna()
            & np.isfinite(paired["ipc"])
            & np.isfinite(paired["baseline_ipc"])
            & (paired["ipc"] > 0)
            & (paired["baseline_ipc"] > 0)
        ]

        if paired.empty:
            gm = np.nan
        else:
            ratios = (
                paired["ipc"].to_numpy(dtype=float)
                / paired["baseline_ipc"].to_numpy(dtype=float)
            )
            gm = float(np.exp(np.log(ratios).mean()))

        rows.append({
            "model": model,
            "ipc_geomean_speedup": gm,
            "speedup_trace_count": len(paired),
        })

    return pd.DataFrame(rows)


# ============================================================
#       Geometric mean of normalized MPKI reductions
# ============================================================

def calculate_geomean_mpki_reduction(df, baseline=BASELINE_MODEL):
    """
    MPKI reduction ratio:
        baseline MPKI / policy MPKI

    GM MPKI reduction:
        exp(mean(log(per-trace MPKI reduction ratios)))

    A ratio above 1 means lower MPKI than the baseline.
    A ratio below 1 means higher MPKI than the baseline.

    Traces with zero, negative, missing, or non-finite MPKI
    for either policy or baseline are excluded. Zero-MPKI
    cases are reported as excluded rather than assigning an
    arbitrary epsilon.
    """
    if "branch_mpki" not in df.columns:
        raise ValueError("Branch MPKI data is required.")

    mpki = df[["model", "trace", "branch_mpki"]].copy()
    mpki["branch_mpki"] = pd.to_numeric(
        mpki["branch_mpki"], errors="coerce"
    )

    # Aggregate duplicate runs of the same model and trace.
    mpki = mpki.groupby(
        ["model", "trace"], as_index=False
    )["branch_mpki"].mean()

    baseline_df = (
        mpki[mpki["model"] == baseline][["trace", "branch_mpki"]]
        .rename(columns={"branch_mpki": "baseline_mpki"})
    )

    if baseline_df.empty:
        raise ValueError(
            f"Baseline model '{baseline}' was not found for MPKI."
        )

    rows = []

    for model in mpki["model"].unique():
        policy_df = mpki[
            mpki["model"] == model
        ][["trace", "branch_mpki"]]

        paired = policy_df.merge(
            baseline_df, on="trace", how="inner"
        )

        valid = (
            paired["branch_mpki"].notna()
            & paired["baseline_mpki"].notna()
            & np.isfinite(paired["branch_mpki"])
            & np.isfinite(paired["baseline_mpki"])
            & (paired["branch_mpki"] > 0)
            & (paired["baseline_mpki"] > 0)
        )

        paired = paired[valid].copy()

        if paired.empty:
            gm = np.nan
        else:
            # Lower MPKI is better, so invert the ratio.
            ratios = (
                paired["baseline_mpki"].to_numpy(dtype=float)
                / paired["branch_mpki"].to_numpy(dtype=float)
            )

            gm = float(np.exp(np.log(ratios).mean()))

        rows.append({
            "model": model,
            "mpki_geomean_reduction": gm,
            "mpki_trace_count": len(paired),
        })

    return pd.DataFrame(rows)


# ============================================================
#               Plot normalized geometric means
# ============================================================

def plot_geomean_ratio(
    data,
    ratio_column,
    ylabel,
    title,
    output,
    reference_label,
):
    data = data.dropna(subset=[ratio_column]).copy()

    if data.empty:
        return False

    order = [m for m in PLOT_ORDER if m in data["model"].unique()]
    order += [m for m in data["model"].unique() if m not in order]

    data["order"] = data["model"].map(
        {model: i for i, model in enumerate(order)}
    )
    data = data.sort_values("order").reset_index(drop=True)

    values = data[ratio_column].astype(float)
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

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=reference_label,
    )

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=50, ha="right")
    ax.set_ylabel(ylabel)
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


def plot_geomean_speedup(data, title, output):
    return plot_geomean_ratio(
        data=data,
        ratio_column="ipc_geomean_speedup",
        ylabel="Geometric mean IPC speedup vs LRU",
        title=title,
        output=output,
        reference_label="LRU baseline = 1.0",
    )


def calculate_per_trace_ipc_speedup(df, baseline=BASELINE_MODEL):
    """Calculate IPC speedup (%) for each model relative to baseline per trace."""
    ipc = df[["model", "trace", "ipc"]].copy()
    ipc["ipc"] = pd.to_numeric(ipc["ipc"], errors="coerce")
    ipc = ipc.groupby(["model", "trace"], as_index=False)["ipc"].mean()

    baseline_df = (
        ipc[ipc["model"] == baseline][["trace", "ipc"]]
        .rename(columns={"ipc": "baseline_ipc"})
    )
    if baseline_df.empty:
        raise ValueError(
            f"Baseline model '{baseline}' was not found. "
            "Cannot calculate per-trace IPC speedup."
        )

    paired = ipc.merge(baseline_df, on="trace", how="inner")
    paired = paired[
        paired["ipc"].notna()
        & paired["baseline_ipc"].notna()
        & np.isfinite(paired["ipc"])
        & np.isfinite(paired["baseline_ipc"])
        & (paired["ipc"] > 0)
        & (paired["baseline_ipc"] > 0)
    ].copy()

    paired["ipc_speedup_pct"] = (
        (paired["ipc"] / paired["baseline_ipc"]) - 1.0
    ) * 100.0
    return paired[["model", "trace", "ipc_speedup_pct"]]



def plot_ipc_all_traces_grouped(df, output):
    """One grouped bar chart: each trace has one IPC bar per BTB policy."""
    data = df[["model", "trace", "ipc"]].copy()
    data["ipc"] = pd.to_numeric(data["ipc"], errors="coerce")
    data = data[data["trace"].isin(TRACES) & data["ipc"].notna()].copy()
    if data.empty:
        return False

    # Average duplicate runs for the same policy/trace pair.
    data = data.groupby(["trace", "model"], as_index=False)["ipc"].mean()
    models = [m for m in PLOT_ORDER if m in data["model"].unique()]
    models += [m for m in data["model"].unique() if m not in models]
    traces = [t for t in TRACES if t in data["trace"].unique()]
    pivot = data.pivot(index="trace", columns="model", values="ipc").reindex(
        index=traces, columns=models
    )

    # Wide layout keeps the model bars distinguishable for every trace.
    fig_width = max(15, len(traces) * 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 7.5))
    x = np.arange(len(traces))
    group_width = 0.86
    bar_width = group_width / max(len(models), 1)

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * bar_width
        values = pivot[model].to_numpy(dtype=float)
        ax.bar(
            x + offset,
            values,
            width=bar_width * 0.94,
            label=DISPLAY_NAMES.get(model, model),
            color=MODEL_STYLE.get(model, ("grey", ""))[0],
            edgecolor="black",
            linewidth=0.35,
            hatch=MODEL_STYLE.get(model, ("grey", ""))[1],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(traces, rotation=25, ha="right")
    ax.set_ylabel("Instructions per cycle (IPC)")
    ax.set_title("Per-trace IPC by BTB policy  all traces in one plot")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        title="BTB policy",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        title_fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True

def plot_per_trace_ipc_speedup(data, trace, output):
    """Plot each policy's IPC change (%) against LRU for one trace."""
    data = data[data["trace"] == trace].dropna(
        subset=["ipc_speedup_pct"]
    ).copy()
    if data.empty:
        return False

    order = [m for m in PLOT_ORDER if m in data["model"].unique()]
    order += [m for m in data["model"].unique() if m not in order]
    data["order"] = data["model"].map(
        {model: i for i, model in enumerate(order)}
    )
    data = data.sort_values("order").reset_index(drop=True)

    values = data["ipc_speedup_pct"].astype(float).to_numpy()
    labels = [DISPLAY_NAMES.get(m, m) for m in data["model"]]
    colors = [MODEL_STYLE.get(m, ("grey", ""))[0] for m in data["model"]]

    fig, ax = plt.subplots(figsize=(max(10, len(data) * 0.8), 5.8))
    bars = ax.bar(
        range(len(data)),
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, model in zip(bars, data["model"]):
        bar.set_hatch(MODEL_STYLE.get(model, ("grey", ""))[1])

    ax.axhline(0, color="black", linewidth=1.1)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("IPC speedup vs LRU (%)")
    ax.set_title(f"Per-trace IPC speedup vs LRU  {trace}")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)

    low = min(float(np.min(values)), 0.0)
    high = max(float(np.max(values)), 0.0)
    span = high - low
    padding = span * 0.18 if span else 0.1
    ax.set_ylim(low - padding, high + padding)
    offset = max(span * 0.018, 0.02)

    for bar, value in zip(bars, values):
        if value >= 0:
            y, va = value + offset, "bottom"
        else:
            y, va = value - offset, "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+.2f}%",
            ha="center",
            va=va,
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_geomean_mpki_reduction(data, title, output):
    return plot_geomean_ratio(
        data=data,
        ratio_column="mpki_geomean_reduction",
        ylabel="Geometric mean MPKI reduction ratio vs LRU",
        title=title,
        output=output,
        reference_label="LRU baseline = 1.0",
    )


def make_plots(
    df,
    metrics,
    plots_dir,
    speedup_summary,
    mpki_summary,
    per_trace_speedup,
):
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

        # Per-trace IPC speedup (%) relative to the LRU/convBTB baseline.
        # Include every policy, including standalone branch-priority variants.
        if group_name == "main":
            speedup_trace_dir = group_dir / "per_trace_speedup"
            speedup_trace_dir.mkdir(parents=True, exist_ok=True)
            for trace in sorted(per_trace_speedup["trace"].dropna().unique()):
                if trace == "unknown":
                    continue
                count += plot_per_trace_ipc_speedup(
                    per_trace_speedup,
                    trace,
                    speedup_trace_dir / f"ipc_speedup_{trace}.png",
                )

        # IPC geometric mean plot for this group.
        group_speedup = speedup_summary[
            speedup_summary["model"].isin(models)
        ]

        count += plot_geomean_speedup(
            group_speedup,
            f"Geometric Mean IPC Speedup vs LRU {group_name}",
            group_dir / "ipc_geomean_speedup.png",
        )

        # MPKI geometric mean plot for this group.
        group_mpki = mpki_summary[
            mpki_summary["model"].isin(models)
        ]

        count += plot_geomean_mpki_reduction(
            group_mpki,
            f"Geometric Mean MPKI Reduction vs LRU {group_name}",
            group_dir / "mpki_geomean_reduction.png",
        )

    # Overall comparison of all policies.
    count += plot_geomean_speedup(
        speedup_summary,
        "Geometric Mean IPC Speedup vs LRU (convBTB)",
        plots_dir / "ipc_geomean_speedup.png",
    )

    count += plot_geomean_mpki_reduction(
        mpki_summary,
        "Geometric Mean MPKI Reduction vs LRU (convBTB)",
        plots_dir / "mpki_geomean_reduction.png",
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

    # Calculate normalized GMs from the complete dataset so each
    # policy is compared against the same baseline trace set.
    if "ipc" not in df or not df["ipc"].notna().any():
        raise SystemExit("No IPC values were parsed; cannot calculate speedup.")

    if "branch_mpki" not in df or not df["branch_mpki"].notna().any():
        raise SystemExit("No branch MPKI values were parsed.")

    try:
        speedup_summary = calculate_geomean_speedup(df)
        mpki_summary = calculate_geomean_mpki_reduction(df)
        per_trace_speedup = calculate_per_trace_ipc_speedup(df)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Merge both normalized GM metrics into the model-level summary.
    summary = summary.merge(
        speedup_summary,
        on="model",
        how="left",
    )

    summary = summary.merge(
        mpki_summary,
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

    mpki_summary.to_csv(
        out / "mpki_geomean_reduction.csv",
        index=False,
        float_format="%.15g",
    )

    per_trace_speedup.to_csv(
        out / "ipc_speedup_per_trace.csv",
        index=False,
        float_format="%.15g",
    )

    plots_dir = out / "plots"

    count = make_plots(
        df,
        present,
        plots_dir,
        speedup_summary,
        mpki_summary,
        per_trace_speedup,
    )

    # Single grouped chart: all traces on the x-axis, with one bar per model.
    plots_dir.mkdir(parents=True, exist_ok=True)
    if plot_ipc_all_traces_grouped(
        df, plots_dir / "ipc_per_trace_all_models.png"
    ):
        count += 1

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
    print(f"All-traces grouped IPC plot: {plots_dir / 'ipc_per_trace_all_models.png'}")
    print(
        "Per-trace IPC speedup plots: "
        f"{plots_dir / 'per_trace_speedup'}"
    )
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

    print("\nGeometric mean MPKI reduction vs LRU:")
    print(
        mpki_summary.sort_values(
            "mpki_geomean_reduction",
            ascending=False,
            na_position="last",
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
