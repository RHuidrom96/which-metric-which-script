"""
experiment4_null_output_validity.py — Experiment 4: Null-Output Validity
─────────────────────────────────────────────────────────────────────────────
Tests floor-score inflation on degenerate PRIMARY submissions (BLEU <= 1.0).
Computes floor-inflation ratio = score(degenerate) / score(top system in pair).

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import config
import stats_utils as su
import plot_utils as pu
from data_loader import load_all_scores, wide_by_instrument

logger = logging.getLogger(__name__)


def run():
    logger.info(f"Executing Experiment 4: Null-Output Validity Testing — WMT{config.YEAR}...")
    df = load_all_scores(level="system")

    degenerate_rows = []
    instruments = config.METRIC_INSTRUMENTS + config.LLM_JUDGE_INSTRUMENTS

    for pair in config.LANGUAGE_PAIRS:
        wide = wide_by_instrument(df, pair)
        if wide.empty or "BLEU" not in wide.columns:
            continue

        deg_systems = wide[wide["BLEU"] <= config.DEGENERATE_BLEU_MAX]
        for _, deg in deg_systems.iterrows():
            for inst in instruments:
                if inst not in wide.columns:
                    continue
                deg_score = deg[inst]
                higher_is_better = config.HIGHER_IS_BETTER.get(inst, True)
                top_score = wide[inst].max() if higher_is_better else wide[inst].min()


                ratio = deg_score / (top_score + 1e-9) if higher_is_better else top_score / (deg_score + 1e-9)

                gap = (top_score - deg_score) if higher_is_better else (deg_score - top_score)

                degenerate_rows.append({
                    "language_pair": pair,
                    "team": deg["team"],
                    "system": deg["system"],
                    "instrument": inst,
                    "degenerate_score": deg_score,
                    "pair_top_score": top_score,
                    "floor_gap": float(gap),
                    "floor_inflation_ratio": float(ratio),
                })

    detail_df = pd.DataFrame(degenerate_rows)
    detail_df.to_csv(f"{config.TABLES_DIR}/exp4_floor_inflation_detail.csv", index=False)

    if detail_df.empty:
        logger.info(f"No primary systems with BLEU <= {config.DEGENERATE_BLEU_MAX} found for WMT{config.YEAR} "
                     f"— writing an empty summary table.")
        summary = pd.DataFrame(columns=["instrument", "mean_floor", "mean_floor_gap",
                                         "mean_floor_inflation_ratio", "ratio_ci_low", "ratio_ci_high",
                                         "max_floor_inflation_ratio"])
    else:
        summary_rows = []
        for inst, grp in detail_df.groupby("instrument"):
            ratio_pt, ratio_lo, ratio_hi = su.bootstrap_ci(grp["floor_inflation_ratio"].values, np.mean)
            summary_rows.append({
                "instrument": inst,
                "mean_floor": grp["degenerate_score"].mean(),
                "mean_floor_gap": grp["floor_gap"].mean(),
                "mean_floor_inflation_ratio": ratio_pt,
                "ratio_ci_low": ratio_lo,
                "ratio_ci_high": ratio_hi,
                "max_floor_inflation_ratio": grp["floor_inflation_ratio"].max(),
            })
        summary = pd.DataFrame(summary_rows).sort_values("mean_floor_inflation_ratio", ascending=False)

    summary.to_csv(f"{config.TABLES_DIR}/exp4_floors_and_inflation_ratios.csv", index=False)

    _plot_floor_inflation(summary)
    _plot_floor_inflation_detail_heatmap(detail_df)

    print("\nExperiment 4: Null-Output Validity Summary:")
    print(summary.to_string(index=False) if not summary.empty else "  (no degenerate primary systems this year)")


def _plot_floor_inflation(summary_df):
    if summary_df.empty:
        return
    plt.figure(figsize=(10, 6))

    plot_df = summary_df.sort_values('mean_floor_inflation_ratio', ascending=True)

    has_ci = {'ratio_ci_low', 'ratio_ci_high'}.issubset(plot_df.columns)
    yerr = None
    if has_ci:
        lo = (plot_df['mean_floor_inflation_ratio'] - plot_df['ratio_ci_low']).clip(lower=0)
        hi = (plot_df['ratio_ci_high'] - plot_df['mean_floor_inflation_ratio']).clip(lower=0)
        yerr = [lo.values, hi.values]

    plt.bar(plot_df['instrument'], plot_df['mean_floor_inflation_ratio'], color='coral', edgecolor='black',
            yerr=yerr, capsize=4, ecolor='black')
    plt.axhline(y=1.0, color='red', linestyle='-', linewidth=2, label='Worst (Scores degenerate = best system)')
    plt.axhline(y=0.0, color='green', linestyle='--', linewidth=2, label='Ideal (Scores degenerate at true 0)')

    plt.title(f'Experiment 4: Null-Output Validity (Floor-Inflation Ratio) — WMT{config.YEAR}', fontweight='bold')
    plt.ylabel('Mean Floor-Inflation Ratio')
    plt.xlabel('Instrument')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}/exp4_floor_inflation.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_floor_inflation_detail_heatmap(detail_df: pd.DataFrame):

    if detail_df.empty:
        return
    matrix = detail_df.pivot_table(index="language_pair", columns="instrument",
                                    values="floor_inflation_ratio", aggfunc="mean")
    matrix = matrix.reindex([p for p in config.LANGUAGE_PAIRS if p in matrix.index])
    pu.annotated_heatmap(
        matrix,
        title=f"Experiment 4: Floor-Inflation Ratio by Language Pair x Instrument — WMT{config.YEAR}",
        out_path=f"{config.FIGURES_DIR}/exp4_floor_inflation_detail_heatmap.png",
        cmap="coolwarm", vmin=0, vmax=1, fmt="{:.2f}",
        cbar_label="Floor-inflation ratio (0=ideal, 1=worst)",
    )
