#!/usr/bin/env python3
"""
Visualize scored outputs from experiments 1–5.

Four research questions:

  Q1 (experiment_2 scored): Which traits cannot be expressed both strongly
     and coherently? Plots trait expression score vs coherence score, with
     a ranked bar chart colored by coherence.

  Q2 (experiment_2 vs experiment_1 scored): Which traits do not transfer
     from persona prompts to prose writing tasks? Requires experiment_1
     outputs to have been scored with the persona judge (see note below).

  Q3 (experiment_3 scored): Which method of combining two trait vectors is
     most successful? LaTeX booktabs table of mean ± SEM scores per method
     (trait 1, trait 2, average, minimum, coherence, and optionally Δ vs
     single-trait baseline from experiment_2). Saved to q3_method_comparison.tex
     and printed to stdout.

  Q4 (experiment_5 scored): How do PPCM and coherence change as more traits
     are combined? Line plots of mean PPCM and mean coherence vs. number of
     combined traits, with one line per combination method (± SEM bands).

  Q5 (experiment_6 signals): Do user demos that exhibit a trait activate its
     steering vector more strongly than demos that do not? For each trait and
     collected layer, plots mean projection of (h_demo − h_prompt) onto the
     steering vector direction, comparing trait-present vs. trait-absent demos
     (sorted by discriminability).

Note for Q2: experiment_1 outputs need persona-judge scores. Run
score_experiment_2.py on the experiment_1 CSVs after renaming (or
symlinking) the task_prompt column to question, or use a copy with that
column added.

Usage (from eval_prose/src/):
    python visualize_experiments.py \\
        --exp2-dir experiments/experiment_2_scored/ \\
        --exp3-dir experiments/experiment_3_scored/ \\
        --output-dir figures/

    # With experiment_1 for Q2, experiment_5 for Q4, experiment_6 for Q5:
    python visualize_experiments.py \\
        --exp2-dir experiments/experiment_2_scored/ \\
        --exp1-dir experiments/experiment_1_scored/ \\
        --exp3-dir experiments/experiment_3_scored/ \\
        --exp5-dir experiments/experiment_5_scored/ \\
        --exp6-csv experiments/experiment_6/experiment_6_signals.csv \\
        --output-dir figures/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Traits present in Llama settings but not Qwen — excluded from all outputs
_EXCLUDE_TRAITS = {"email_epithet_signoff", "open_with_movie_ref", "semicolon_usage"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_dir(directory: str, pattern: str = "*_scored.csv") -> pd.DataFrame | None:
    """Load and concatenate all CSVs matching pattern in directory."""
    if directory is None:
        return None
    paths = list(Path(directory).glob(pattern))
    if not paths:
        print(f"  No files matching '{pattern}' in {directory}")
        return None
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df)} rows from {len(paths)} file(s) in {directory}")
    return df


def _task_type_from_stem(stem: str) -> str:
    stem = stem.lower()
    if "email" in stem:
        return "email_writing"
    if "summ" in stem or "summarization" in stem:
        return "summarization"
    return "unknown"


def _load_exp1_dir(directory: str, pattern: str = "*_scored.csv") -> pd.DataFrame | None:
    """Like _load_dir but adds a task_type column inferred from each filename."""
    if directory is None:
        return None
    paths = list(Path(directory).glob(pattern))
    if not paths:
        print(f"  No files matching '{pattern}' in {directory}")
        return None
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["task_type"] = _task_type_from_stem(p.stem)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df)} rows from {len(paths)} file(s) in {directory}")
    return df


TASK_COLORS = {"email_writing": "#1f77b4", "summarization": "#ff7f0e"}
TASK_LABELS = {"email_writing": "Email writing", "summarization": "Summarization"}


def _sem(x):
    return x.std() / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _trait_label(name: str) -> str:
    return name.replace("_", " ")


# ── Q1 ────────────────────────────────────────────────────────────────────────

def plot_q1(df: pd.DataFrame, output_dir: Path, trait_thresh: float, coh_thresh: float, fmt: str):
    """Scatter + ranked bar: which traits are strong and coherent?"""

    stats = (
        df.groupby("trait", sort=False)
        .agg(
            ts_mean=("trait_score", "mean"),
            ts_sem=("trait_score", _sem),
            coh_mean=("coherence_score", "mean"),
            coh_sem=("coherence_score", _sem),
        )
        .reset_index()
        .dropna(subset=["ts_mean"])
        .sort_values("ts_mean", ascending=False)
    )

    def quad_color(ts, cs):
        if ts >= trait_thresh and cs >= coh_thresh:
            return "#2ca02c"   # green  – both good
        if ts >= trait_thresh:
            return "#ff7f0e"   # orange – strong but incoherent
        if cs >= coh_thresh:
            return "#1f77b4"   # blue   – coherent but weak
        return "#d62728"       # red    – both weak

    point_colors = [quad_color(r.ts_mean, r.coh_mean) for _, r in stats.iterrows()]

    fig, (ax_sc, ax_bar) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: scatter
    ax_sc.scatter(stats["ts_mean"], stats["coh_mean"], c=point_colors, s=90, zorder=3)
    ax_sc.errorbar(
        stats["ts_mean"], stats["coh_mean"],
        xerr=stats["ts_sem"], yerr=stats["coh_sem"],
        fmt="none", color="gray", alpha=0.35, zorder=2,
    )
    for _, r in stats.iterrows():
        ax_sc.annotate(
            _trait_label(r["trait"]),
            (r.ts_mean, r.coh_mean),
            xytext=(5, 3), textcoords="offset points",
            fontsize=11, alpha=0.9,
        )
    ax_sc.axvline(trait_thresh, ls="--", color="gray", alpha=0.45, lw=1)
    ax_sc.axhline(coh_thresh,   ls="--", color="gray", alpha=0.45, lw=1)
    ax_sc.set_xlim(0, 100)
    ax_sc.set_ylim(0, 100)
    ax_sc.set_xlabel("Mean trait expression score", fontsize=15)
    ax_sc.set_ylabel("Mean coherence score", fontsize=15)
    ax_sc.set_title("Trait expression vs coherence", fontsize=16)
    ax_sc.grid(True, alpha=0.25)
    legend_patches = [
        mpatches.Patch(color="#2ca02c", label=f"Strong + coherent (≥{trait_thresh:.0f}, ≥{coh_thresh:.0f})"),
        mpatches.Patch(color="#ff7f0e", label=f"Strong but incoherent (coh < {coh_thresh:.0f})"),
        mpatches.Patch(color="#1f77b4", label=f"Coherent but weak (trait < {trait_thresh:.0f})"),
        mpatches.Patch(color="#d62728", label="Both weak"),
    ]
    ax_sc.legend(handles=legend_patches, fontsize=12, loc="lower right")

    # Right: horizontal bar chart ranked by trait score, colored by coherence
    sorted_stats = stats.sort_values("ts_mean")
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = cm.RdYlGn
    bar_colors = [cmap(norm(c)) for c in sorted_stats["coh_mean"]]

    y = np.arange(len(sorted_stats))
    ax_bar.barh(
        y, sorted_stats["ts_mean"],
        xerr=sorted_stats["ts_sem"],
        color=bar_colors, edgecolor="white", linewidth=0.4,
        capsize=2,
    )
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([_trait_label(t) for t in sorted_stats["trait"]], fontsize=12)
    ax_bar.set_ylim(-0.5, len(sorted_stats) - 0.5)
    ax_bar.axvline(trait_thresh, ls="--", color="gray", alpha=0.45, lw=1)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Mean trait expression score", fontsize=15)
    ax_bar.set_title("Traits ranked by expression", fontsize=16)
    ax_bar.grid(True, alpha=0.25, axis="x")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_bar, label="Coherence score", shrink=0.6, pad=0.02)

    fig.tight_layout()
    out = output_dir / f"q1_strength_vs_coherence.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Q2 ────────────────────────────────────────────────────────────────────────

def plot_q2(
    df_exp2: pd.DataFrame,
    df_exp1: pd.DataFrame,
    output_dir: Path,
    fmt: str,
):
    """Scatter + gap bar: which traits transfer from persona prompts to prose tasks?"""

    # Persona judge (exp2): 0–100 → normalise to 0–100%
    df_exp2 = df_exp2.copy()
    df_exp2["trait_score_pct"] = df_exp2["trait_score"] / 100 * 100  # identity, kept explicit

    exp2 = (
        df_exp2.groupby("trait")
        .agg(ts_mean=("trait_score_pct", "mean"), ts_sem=("trait_score_pct", _sem))
        .reset_index()
        .dropna(subset=["ts_mean"])
    )

    # Prose PPCM (exp1): range is -2 to 2 → normalise to 0–100%
    if "ppcm" not in df_exp1.columns and "trait_score" not in df_exp1.columns:
        print("  Q2: experiment_1 data has no 'ppcm' or 'trait_score' column. Skipping.")
        return

    df_exp1 = df_exp1.copy()
    score_col = "ppcm" if "ppcm" in df_exp1.columns else "trait_score"
    PPCM_MIN, PPCM_MAX = -2.0, 2.0
    df_exp1["trait_score_pct"] = (df_exp1[score_col] - PPCM_MIN) / (PPCM_MAX - PPCM_MIN) * 100

    has_task = "task_type" in df_exp1.columns
    group_cols = ["trait", "task_type"] if has_task else ["trait"]

    exp1 = (
        df_exp1.groupby(group_cols)
        .agg(ts_mean=("trait_score_pct", "mean"), ts_sem=("trait_score_pct", _sem))
        .reset_index()
        .dropna(subset=["ts_mean"])
    )

    merged = exp2.merge(exp1, on="trait", suffixes=("_e2", "_e1"))
    if merged.empty:
        print("  Q2: No overlapping traits between exp1 and exp2. Skipping.")
        return

    fig, (ax_sc, ax_gap) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: scatter exp2 vs exp1, coloured by task type
    lim = [0, 105]
    if has_task:
        for task_type, grp in merged.groupby("task_type"):
            color = TASK_COLORS.get(task_type, "gray")
            label = TASK_LABELS.get(task_type, task_type)
            ax_sc.scatter(grp["ts_mean_e2"], grp["ts_mean_e1"], s=80, zorder=3,
                          color=color, label=label)
            ax_sc.errorbar(
                grp["ts_mean_e2"], grp["ts_mean_e1"],
                xerr=grp["ts_sem_e2"], yerr=grp["ts_sem_e1"],
                fmt="none", color=color, alpha=0.35, zorder=2,
            )
            for _, r in grp.iterrows():
                ax_sc.annotate(
                    _trait_label(r["trait"]),
                    (r.ts_mean_e2, r.ts_mean_e1),
                    xytext=(5, 3), textcoords="offset points", fontsize=11,
                )
    else:
        ax_sc.scatter(merged["ts_mean_e2"], merged["ts_mean_e1"], s=80, zorder=3, color="#5a5ea8")
        ax_sc.errorbar(
            merged["ts_mean_e2"], merged["ts_mean_e1"],
            xerr=merged["ts_sem_e2"], yerr=merged["ts_sem_e1"],
            fmt="none", color="gray", alpha=0.35, zorder=2,
        )
        for _, r in merged.iterrows():
            ax_sc.annotate(
                _trait_label(r["trait"]),
                (r.ts_mean_e2, r.ts_mean_e1),
                xytext=(5, 3), textcoords="offset points", fontsize=11,
            )

    ax_sc.plot(lim, lim, ls="--", color="gray", alpha=0.45, lw=1, label="Perfect transfer")
    ax_sc.set_xlim(lim)
    ax_sc.set_ylim(lim)
    ax_sc.set_xlabel("Persona judge score — % of range [0, 100]", fontsize=15)
    ax_sc.set_ylabel("Prose judge score — % of range [−2, 2]", fontsize=15)
    ax_sc.set_title("Persona-prompt expression vs prose-task expression", fontsize=16)
    ax_sc.legend(fontsize=13)
    ax_sc.grid(True, alpha=0.25)

    # Right: transfer gap bar chart, coloured by task type
    merged["gap"] = merged["ts_mean_e2"] - merged["ts_mean_e1"]
    merged_s = merged.sort_values("gap", ascending=False)
    y = np.arange(len(merged_s))

    if has_task:
        bar_colors = [TASK_COLORS.get(t, "gray") for t in merged_s["task_type"]]
        ylabels = [
            f"{_trait_label(r.trait)}"
            for _, r in merged_s.iterrows()
        ]
        task_patches = [
            mpatches.Patch(color=TASK_COLORS[t], label=TASK_LABELS[t])
            for t in TASK_COLORS if t in merged_s["task_type"].values
        ]
        ax_gap.legend(handles=task_patches, fontsize=13)
    else:
        bar_colors = ["#d62728" if g > 0 else "#2ca02c" for g in merged_s["gap"]]
        ylabels = [_trait_label(t) for t in merged_s["trait"]]

    ax_gap.barh(y, merged_s["gap"], color=bar_colors, edgecolor="white", linewidth=0.4)
    ax_gap.set_yticks(y)
    ax_gap.set_yticklabels(ylabels, fontsize=12)
    ax_gap.set_ylim(-0.5, len(merged_s) - 0.5)
    ax_gap.axvline(0, color="black", lw=0.8)
    ax_gap.set_xlabel("Gap (persona % − prose %)", fontsize=14)
    title = "Transfer gap per trait" if has_task else "Transfer gap per trait\n(red = weaker on prose tasks)"
    ax_gap.set_title(title, fontsize=16)
    ax_gap.grid(True, alpha=0.25, axis="x")

    fig.tight_layout()
    out = output_dir / f"q2_transfer.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Q3 ────────────────────────────────────────────────────────────────────────

_METHOD_DISPLAY = {
    "orthogonalize":   "Orthogonalize",
    "different_layers": "Diff. layers",
    "tuned_mean":      "Tuned mean",
}

_METHOD_ORDER = [
    "Orthogonalize",
    "Diff. layers",
    "Tuned mean",
    "Unit norm (layer=16, α=4)",
    "Unit norm (layer=20, α=8)",
    "Unit norm (α=10)",
    "Unit norm (α=20)",
    "Unit norm (layer=16, α=30)",
    "Unit norm (layer=20, α=58)",
]


def _method_label(row) -> str:
    if row["method"] == "unit_norm_mean":
        alpha = row.get("alpha")
        if pd.notna(alpha):
            return f"Unit norm (layer={16 if alpha == 4 or alpha == 30 else 20}, α={alpha:.0f})"
    return _METHOD_DISPLAY.get(row["method"], row["method"])


# Reverse map: (alpha, subset_size) → layer for variable-alpha runs.
# Alpha alone is ambiguous for Llama (6.0 and 8.0 appear in both layers), but
# the combination with subset_size is always unique across both model families.
_ALPHA_SUBSET_TO_LAYER: dict[tuple[float, int], int] = {}
for _layer, _alphas in {16: [20.0, 30.0, 40.0, 50.0], 20: [48.0, 58.0, 68.0, 78.0]}.items():  # Qwen
    for _size, _a in enumerate(_alphas, 1):
        _ALPHA_SUBSET_TO_LAYER[(_a, _size)] = _layer
for _layer, _alphas in {16: [2.0, 4.0, 6.0, 8.0], 20: [6.0, 8.0, 10.0, 12.0]}.items():  # Llama
    for _size, _a in enumerate(_alphas, 1):
        _ALPHA_SUBSET_TO_LAYER[(_a, _size)] = _layer

# Fallback map for old fixed-alpha runs (same alpha used for all subset sizes).
# Matches the original _method_label rule: alpha in {4, 30} → layer 16, else → layer 20.
_OLD_ALPHA_TO_LAYER: dict[float, int] = {4.0: 16, 30.0: 16, 8.0: 20, 10.0: 20, 20.0: 20, 58.0: 20}


def _method_label_q4(row) -> str:
    """Like _method_label but collapses unit_norm_mean variants to their layer."""
    if row["method"] == "unit_norm_mean":
        is_v2 = bool(row.get("_is_v2", False))
        # 1. alpha_layer column written by the updated experiment_5.py
        layer = row.get("alpha_layer")
        if pd.isna(layer):
            alpha = row.get("alpha")
            subset_size = row.get("subset_size")
            if pd.notna(alpha):
                # 2. v2: variable-alpha, (alpha, subset_size) is unambiguous
                if is_v2 and pd.notna(subset_size):
                    layer = _ALPHA_SUBSET_TO_LAYER.get((float(alpha), int(subset_size)))
                # 3. Old fixed-alpha: same alpha used for every subset size
                if layer is None:
                    layer = _OLD_ALPHA_TO_LAYER.get(float(alpha))
        if layer is not None and pd.notna(layer):
            prefix = "Unit norm v2" if is_v2 else "Unit norm"
            return f"{prefix} (layer {int(layer)})"
    return _METHOD_DISPLAY.get(row["method"], row["method"])


def plot_q3(df: pd.DataFrame, output_dir: Path, fmt: str, df_exp1: pd.DataFrame | None = None, df_exp2: pd.DataFrame | None = None, model_name: str | None = None):
    """LaTeX booktabs table: which combination method works best?"""

    df = df.copy()
    df["method_label"] = df.apply(_method_label, axis=1)
    df["avg_trait_score"] = (df["trait1_score"] + df["trait2_score"]) / 2
    df["min_trait_score"] = df[["trait1_score", "trait2_score"]].min(axis=1)

    present_methods = df["method_label"].unique()
    method_order = [m for m in _METHOD_ORDER if m in present_methods]
    method_order += [m for m in present_methods if m not in method_order]

    stats = (
        df.groupby("method_label", sort=False)
        .agg(
            t1_mean=("trait1_score",    "mean"),
            t1_sem =("trait1_score",    _sem),
            t2_mean=("trait2_score",    "mean"),
            t2_sem =("trait2_score",    _sem),
            coh_mean=("coherence_score", "mean"),
            coh_sem =("coherence_score", _sem),
            avg_mean=("avg_trait_score", "mean"),
            avg_sem =("avg_trait_score", _sem),
            min_mean=("min_trait_score", "mean"),
            min_sem =("min_trait_score", _sem),
        )
        .reindex(method_order)
        .reset_index()
        .dropna(subset=["t1_mean"])
    )

    has_baseline = False
    if df_exp2 is not None and "trait_score" in df_exp2.columns:
        baseline = df_exp2.groupby("trait")["trait_score"].mean().to_dict()
        reductions = []
        for _, row in df.iterrows():
            for t_col, score_col_e3 in [("trait1", "trait1_score"), ("trait2", "trait2_score")]:
                trait = row[t_col]
                if trait in baseline and pd.notna(row[score_col_e3]):
                    reductions.append({
                        "method_label": row["method_label"],
                        "reduction": row[score_col_e3] - baseline[trait],
                    })
        if reductions:
            red_df = pd.DataFrame(reductions)
            red_stats = (
                red_df.groupby("method_label", sort=False)
                .agg(red_mean=("reduction", "mean"), red_sem=("reduction", _sem))
                .reindex(method_order)
                .reset_index()
            )
            stats = stats.merge(red_stats, on="method_label", how="left")
            has_baseline = True

    def _cell(mean, sem):
        return f"${mean:.1f}{{\\,{{\\scriptstyle\\pm {sem:.1f}}}}}$"

    def _latex_method(label: str) -> str:
        return label.replace("α", r"$\alpha$")

    # Column order: [model,] method, trait1, trait2, avg, [baseline,] coherence
    has_model = model_name is not None
    header = []
    col_fmt = ""
    if has_model:
        header.append("")
        col_fmt += "l"
    header += ["Method", "Trait 1", "Trait 2", "Average"]
    col_fmt += "lrrr"
    if has_baseline:
        header.append(r"$\Delta$ vs baseline")
        col_fmt += "r"
    header.append("Coherence")
    col_fmt += "r"

    n_rows = len(stats)
    model_cell = (
        r"\multirow{" + str(n_rows) + r"}{*}{\rotatebox{90}{\textbf{"
        + model_name + r"}}}"
    ) if has_model else None

    lines = [
        r"% Requires: \usepackage{booktabs,multirow,graphicx}",
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Combination method comparison (mean $\pm$ SEM). "
            r"Trait and coherence scores are 0--100. "
            + (r"$\Delta$ vs baseline is the signed difference from the "
               r"single-trait Experiment~2 score." if has_baseline else "")
            + r"}"
        ),
        r"\label{tab:q3_method_comparison}",
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    for i, (_, row) in enumerate(stats.iterrows()):
        cells = []
        if has_model:
            cells.append(model_cell if i == 0 else "")
        cells += [
            _latex_method(row["method_label"]),
            _cell(row["t1_mean"],  row["t1_sem"]),
            _cell(row["t2_mean"],  row["t2_sem"]),
            _cell(row["avg_mean"], row["avg_sem"]),
        ]
        if has_baseline:
            cells.append(
                _cell(row["red_mean"], row["red_sem"])
                if pd.notna(row.get("red_mean")) else "---"
            )
        cells.append(_cell(row["coh_mean"], row["coh_sem"]))
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    print(latex)

    out = output_dir / "q3_method_comparison.tex"
    out.write_text(latex)
    print(f"  Saved: {out}")


# ── Q5 ────────────────────────────────────────────────────────────────────────

def plot_q5(df: pd.DataFrame, output_dir: Path, fmt: str,
            best_settings: dict | None = None) -> None:
    """
    Horizontal bar chart: difference in mean proj(h_demo − h_prompt, v̂_t)
    between trait-present and trait-absent demos.

    For each trait, only the layer from best_settings is used (falls back to
    the first available layer when the trait is absent from best_settings).
    Bars are green for positive differences and red for negative.
    """
    df = df.copy()
    required = {"trait", "layer", "trait_present", "proj_demo"}
    missing = required - set(df.columns)
    if missing:
        print(f"  Q5: missing columns {missing}. Skipping.")
        return

    best_settings = best_settings or {}
    layers_in_data = sorted(df["layer"].unique().astype(int))

    # Mean ± SEM per (trait, layer, trait_present)
    agg = (
        df.groupby(["trait", "layer", "trait_present"])
        .agg(mean=("proj_demo", "mean"), sem=("proj_demo", _sem))
        .reset_index()
    )

    # For each trait, pick its best layer; fall back to first available
    traits = df["trait"].unique().tolist()
    trait_layer: dict[str, int] = {}
    for t in traits:
        preferred = int(best_settings[t]["layer"]) if t in best_settings else None
        if preferred is not None and preferred in layers_in_data:
            trait_layer[t] = preferred
        else:
            available = sorted(agg[agg["trait"] == t]["layer"].unique().astype(int))
            trait_layer[t] = available[0] if available else layers_in_data[0]

    # Compute difference (present − absent) per trait at its chosen layer
    records = []
    for t in traits:
        layer = trait_layer[t]
        sub = agg[(agg["trait"] == t) & (agg["layer"] == layer)]
        row_p = sub[sub["trait_present"]]
        row_a = sub[~sub["trait_present"]]
        if row_p.empty or row_a.empty:
            continue
        mean_p, sem_p = row_p["mean"].iloc[0], row_p["sem"].iloc[0]
        mean_a, sem_a = row_a["mean"].iloc[0], row_a["sem"].iloc[0]
        diff = mean_p - mean_a
        diff_se = np.sqrt(sem_p ** 2 + sem_a ** 2)
        records.append({"trait": t, "layer": layer, "diff": diff, "se": diff_se})

    if not records:
        print("  Q5: no data after layer selection. Skipping.")
        return

    diff_df = pd.DataFrame(records).sort_values("diff")
    trait_order = diff_df["trait"].tolist()
    n_traits = len(trait_order)

    fig_h = max(5, n_traits * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    y = np.arange(n_traits)
    diffs = diff_df["diff"].values
    ses   = diff_df["se"].values
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in diffs]

    ax.barh(y, diffs, xerr=ses, height=0.6, color=colors, capsize=3)
    ax.set_yticks(y)
    # Show trait label with its layer in parentheses
    ylabels = [
        f"{_trait_label(t)}  (L{diff_df[diff_df['trait']==t]['layer'].iloc[0]})"
        for t in trait_order
    ]
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_ylim(-0.5, n_traits - 0.5)
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel(
        r"$\Delta$ proj$(h_\mathrm{demo} - h_\mathrm{prompt},\;\hat{v}_t)$"
        r"$\;$ [present $-$ absent]",
        fontsize=14,
    )
    ax.set_title(
        "Steering vector activation: trait-present minus trait-absent\n"
        "(each trait at its tuned layer; sorted by difference)",
        fontsize=15,
    )
    ax.grid(True, alpha=0.25, axis="x")

    fig.tight_layout()
    _save(fig, output_dir / f"q5_contrastive.{fmt}")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Q4 ────────────────────────────────────────────────────────────────────────

def plot_q4(df: pd.DataFrame, output_dir: Path, fmt: str):
    """Line plots of PPCM and coherence vs. number of combined traits, by method."""

    df = df.copy()
    df["method_label"] = df.apply(_method_label_q4, axis=1)

    subset_sizes = sorted(df["subset_size"].dropna().unique().astype(int))
    present = df["method_label"].unique().tolist()
    unit_norm = sorted(
        [m for m in present if m.startswith("Unit norm")],
        key=lambda m: (
            int(m.split("layer ")[1].rstrip(")")) if "layer " in m else 0,
            m,  # stable secondary sort by full name (distinguishes v2 from non-v2)
        ),
    )
    base = [m for m in _METHOD_ORDER if m in present and not m.startswith("Unit norm")]
    other = [m for m in present if m not in base and m not in unit_norm]
    method_order = base + unit_norm + other

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_colors = {m: prop_cycle[i % len(prop_cycle)] for i, m in enumerate(method_order)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, metric, ylabel in [
        (axes[0], "ppcm",           "PPCM"),
        (axes[1], "coherence_score","Coherence"),
    ]:
        for method in method_order:
            mdf = df[df["method_label"] == method]
            if mdf.empty or metric not in mdf.columns:
                continue
            agg = (
                mdf.groupby("subset_size")
                .agg(mean=(metric, "mean"), sem=(metric, _sem))
                .reindex(subset_sizes)
                .dropna(subset=["mean"])
            )
            if agg.empty:
                continue
            color = method_colors[method]
            ax.plot(agg.index, agg["mean"], marker="o", label=method, color=color)
            ax.fill_between(
                agg.index,
                agg["mean"] - agg["sem"],
                agg["mean"] + agg["sem"],
                alpha=0.15,
                color=color,
            )
        ax.set_xlabel("Number of combined traits", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_ylim([-2, 2] if metric == 'ppcm' else [0, 100])
        ax.set_xticks(subset_sizes)
        if metric == "ppcm":
            ax.legend(fontsize=13)
        ax.grid(True, alpha=0.25)

    axes[0].set_title("PPCM vs. number of combined traits", fontsize=15)
    axes[1].set_title("Coherence vs. number of combined traits", fontsize=15)

    fig.tight_layout()
    out = output_dir / f"q4_scores_by_num_traits.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--exp1-dir", default=None,
                        help="Dir with scored experiment_1 CSVs (for Q2, optional)")
    parser.add_argument("--exp2-dir", default=None,
                        help="Dir with scored experiment_2 CSVs (for Q1 and Q2)")
    parser.add_argument("--exp3-dir", default=None,
                        help="Dir with scored experiment_3 CSVs (for Q3)")
    parser.add_argument("--exp5-dir", default=None, nargs="+",
                        help="Dir(s) with scored experiment_5 CSVs (for Q4); multiple dirs are concatenated")
    parser.add_argument("--exp6-csv", default=None,
                        help="experiment_6_signals.csv from experiment_6.py (for Q5)")
    parser.add_argument("--best-json", default=None,
                        help="JSON with per-trait {layer, coeff} from tune_steering.py "
                             "(used by Q5 to select the best layer per trait)")
    parser.add_argument("--output-dir", default="figures",
                        help="Directory to save figures (default: figures/)")
    parser.add_argument("--trait-threshold", type=float, default=60.0,
                        help="Trait expression threshold for Q1 quadrant coloring (default: 60)")
    parser.add_argument("--coherence-threshold", type=float, default=75.0,
                        help="Coherence threshold for Q1 quadrant coloring (default: 75)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Output figure format (default: png)")
    parser.add_argument("--model-name", default=None,
                        help="Model name shown as rotated bold label on the left of the Q3 table")
    args = parser.parse_args()

    if (args.exp1_dir is None and args.exp2_dir is None and args.exp3_dir is None
            and not args.exp5_dir and args.exp6_csv is None):
        parser.error(
            "Provide at least one of --exp1-dir, --exp2-dir, --exp3-dir, "
            "--exp5-dir, --exp6-csv"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_exp2 = df_exp1 = df_exp3 = df_exp5 = df_exp6 = None

    if args.exp2_dir:
        print(f"\nLoading experiment_2 data from {args.exp2_dir} ...")
        df_exp2 = _load_dir(args.exp2_dir)
        if df_exp2 is not None:
            df_exp2 = df_exp2[~df_exp2["trait"].isin(_EXCLUDE_TRAITS)]

    if args.exp1_dir:
        print(f"\nLoading experiment_1 data from {args.exp1_dir} ...")
        df_exp1 = _load_exp1_dir(args.exp1_dir)
        if df_exp1 is not None:
            df_exp1 = df_exp1[~df_exp1["trait"].isin(_EXCLUDE_TRAITS)]

    if args.exp3_dir:
        print(f"\nLoading experiment_3 data from {args.exp3_dir} ...")
        df_exp3 = _load_dir(args.exp3_dir)
        if df_exp3 is not None:
            df_exp3 = df_exp3[
                ~df_exp3["trait1"].isin(_EXCLUDE_TRAITS)
                & ~df_exp3["trait2"].isin(_EXCLUDE_TRAITS)
            ]

    if args.exp5_dir:
        dfs5 = []
        for d in args.exp5_dir:
            print(f"\nLoading experiment_5 data from {d} ...")
            _df = _load_dir(d)
            if _df is not None:
                _df["_is_v2"] = "v2" in Path(d).name
                dfs5.append(_df)
        if dfs5:
            df_exp5 = pd.concat(dfs5, ignore_index=True)
            df_exp5 = df_exp5[
                ~df_exp5["traits"].astype(str).apply(
                    lambda s: any(t in _EXCLUDE_TRAITS for t in (x.strip() for x in s.split(";")))
                )
            ]

    best_settings: dict = {}
    if args.best_json:
        import json
        best_json_path = Path(args.best_json)
        if best_json_path.exists():
            with open(best_json_path) as f:
                best_settings = json.load(f)
            print(f"\nLoaded best settings for {len(best_settings)} traits from {args.best_json}")
        else:
            print(f"\nWarning: --best-json not found: {args.best_json}")

    if args.exp6_csv:
        exp6_path = Path(args.exp6_csv)
        if exp6_path.exists():
            print(f"\nLoading experiment_6 signals from {args.exp6_csv} ...")
            df_exp6 = pd.read_csv(exp6_path)
            df_exp6 = df_exp6[~df_exp6["trait"].isin(_EXCLUDE_TRAITS)]
            print(f"  Loaded {len(df_exp6)} rows")
        else:
            print(f"\nWarning: --exp6-csv not found: {args.exp6_csv}")

    any_output = False

    if df_exp2 is not None:
        print("\n── Q1: Trait expression vs coherence ──")
        plot_q1(df_exp2, output_dir, args.trait_threshold, args.coherence_threshold, args.format)
        any_output = True

    if df_exp2 is not None and df_exp1 is not None:
        print("\n── Q2: Persona → prose transfer ──")
        plot_q2(df_exp2, df_exp1, output_dir, args.format)
        any_output = True
    elif df_exp1 is not None and df_exp2 is None:
        print("\nQ2 skipped: --exp2-dir required alongside --exp1-dir")
    elif df_exp2 is not None and df_exp1 is None:
        print("\nQ2 skipped: --exp1-dir not provided (see docstring for how to score experiment_1 outputs)")

    if df_exp3 is not None:
        print("\n── Q3: Combination method comparison ──")
        plot_q3(df_exp3, output_dir, args.format, df_exp1=df_exp1, df_exp2=df_exp2, model_name=args.model_name)
        any_output = True

    if df_exp5 is not None:
        print("\n── Q4: PPCM and coherence vs. number of combined traits ──")
        plot_q4(df_exp5, output_dir, args.format)
        any_output = True

    if df_exp6 is not None:
        print("\n── Q5: Activation signals vs. tuned layer / coefficient ──")
        plot_q5(df_exp6, output_dir, args.format, best_settings=best_settings)
        any_output = True

    if any_output:
        print(f"\nAll figures saved to {output_dir}/")
    else:
        print("\nNo figures produced — check that input directories contain *_scored.csv files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
