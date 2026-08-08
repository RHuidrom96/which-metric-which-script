"""
experiment3_script_sensitivity.py — Experiment 3: Script Sensitivity & Tiers
─────────────────────────────────────────────────────────────────────────────
Evaluates Manipuri dual-script score shifts (Meitei Mayek - Bengali) and tests
monotonic neural metric inflation across coverage tiers 1-4.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import config
import stats_utils as su
import plot_utils as pu
from data_loader import load_all_scores, wide_by_instrument, oriented

logger = logging.getLogger(__name__)


def run():
    logger.info(f"Executing Experiment 3: Script Sensitivity & Coverage Tiers — WMT{config.YEAR}...")
    df = load_all_scores(level="system")

    # 1. Paired Script Shift (Manipuri)
    shift_rows = []
    instruments = config.METRIC_INSTRUMENTS + config.LLM_JUDGE_INSTRUMENTS

    if not config.SCRIPT_PAIR_DIRECTIONS:
        logger.warning(f"WMT{config.YEAR} has no dual-script Manipuri pairs — "
                        f"the paired script-shift table cannot be computed for this year. "
                        f"Writing an empty table so downstream code has a stable file to read.")

    for spec in config.SCRIPT_PAIR_DIRECTIONS:
        mtei = wide_by_instrument(df, spec["meitei_pair"])
        beng = wide_by_instrument(df, spec["bengali_pair"])

        if mtei.empty or beng.empty:
            logger.warning(f"Skipping script-pair '{spec['direction']}': "
                            f"'{spec['meitei_pair']}' or '{spec['bengali_pair']}' not present "
                            f"in the loaded data — check config.LANGUAGE_PAIRS / the results CSV.")
            continue

        merged = mtei.merge(beng, on="team", suffixes=("_mtei", "_beng"))
        if merged.empty:
            logger.warning(f"Skipping script-pair '{spec['direction']}': no teams submitted to both scripts.")
            continue

        for inst in instruments:
            col_m, col_b = f"{inst}_mtei", f"{inst}_beng"
            if col_m not in merged.columns or col_b not in merged.columns:
                continue

            shifts = merged[col_m] - merged[col_b]
            pt, lo, hi = su.bootstrap_ci(shifts.values, np.mean)

            shift_rows.append({
                "direction": spec["direction"],
                "instrument": inst,
                "mean_shift_meitei_minus_bengali": pt,
                "ci_low": lo,
                "ci_high": hi,
                "n_paired_systems": len(merged),
            })

    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(f"{config.TABLES_DIR}/exp3_paired_script_shifts.csv", index=False)
    _plot_script_shift(shift_df)

    # 2. Coverage Tier Inflation Extension
    df_o = oriented(df)
    tier_rows = []
    for pair in config.LANGUAGE_PAIRS:
        tier = config.tier_for_pair(pair)
        wide = wide_by_instrument(df_o, pair)
        if wide.empty or tier is None:
            continue

        for inst in instruments:
            if inst not in wide.columns:
                continue
            vals = wide[inst].dropna().values
            if len(vals) < 2:
                continue

            top_minus_median = np.max(vals) - np.median(vals)
            rng = np.max(vals) - np.min(vals)

            infl_pt, infl_lo, infl_hi = su.bootstrap_ci(
                vals, lambda x: np.max(x) - np.median(x)
            )
            rng_pt, rng_lo, rng_hi = su.bootstrap_ci(
                vals, lambda x: np.max(x) - np.min(x)
            )

            tier_rows.append({
                "language_pair": pair,
                "tier": tier,
                "instrument": inst,
                "n_systems": len(vals),
                # Inflation: top system vs. the pack's median. Independent
                # of range by construction.
                "inflation_top_minus_median": infl_pt,
                "inflation_ci_low": infl_lo,
                "inflation_ci_high": infl_hi,

                "range": rng_pt,
                "range_ci_low": rng_lo,
                "range_ci_high": rng_hi,
   
                "range_normalised_gap": top_minus_median / (rng + 1e-9),
            })

    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(f"{config.TABLES_DIR}/exp3_inflation_by_tier.csv", index=False)

    _plot_tier_inflation(tier_df)

    print("\nExperiment 3: Script Sensitivity Summary:")
    if shift_df.empty:
        print(f"  (no paired-script-shift rows for WMT{config.YEAR} — see log above)")
    else:
        print(shift_df.to_string(index=False))


def _plot_script_shift(shift_df: pd.DataFrame):

    if shift_df.empty:
        logger.info(f"No paired script-shift rows for WMT{config.YEAR} — skipping exp3_paired_script_shifts.png.")
        return
    pu.grouped_bar(
        shift_df, x="instrument", y="mean_shift_meitei_minus_bengali", group="direction",
        yerr_low="ci_low", yerr_high="ci_high",
        title=f"Experiment 3: Meitei-Mayek minus Bengali Script Shift — WMT{config.YEAR}",
        ylabel="Mean Shift (Meitei - Bengali)",
        out_path=f"{config.FIGURES_DIR}/exp3_paired_script_shifts.png",
    )


def _plot_tier_inflation(tier_df):

    if tier_df.empty:
        logger.warning("No coverage-tier rows produced — skipping exp3 tier figures.")
        return

    _line_and_heatmap(
        tier_df, value_col="inflation_top_minus_median",
        line_ylabel="Top - Median (Inflation)",
        line_title=f"Neural Metric Inflation across Coverage Tiers — WMT{config.YEAR}",
        line_out=f"{config.FIGURES_DIR}/exp3_inflation_by_tier.png",
        heat_title=f"Inflation (Top - Median) by Tier x Instrument — WMT{config.YEAR}",
        heat_out=f"{config.FIGURES_DIR}/exp3_inflation_by_tier_heatmap.png",
        cbar_label="Top - median",
    )

    _line_and_heatmap(
        tier_df, value_col="range",
        line_ylabel="Score Range (Max - Min)",
        line_title=f"Neural Metric Range Compression across Coverage Tiers — WMT{config.YEAR}",
        line_out=f"{config.FIGURES_DIR}/exp3_range_by_tier.png",
        heat_title=f"Score Range by Tier x Instrument — WMT{config.YEAR}",
        heat_out=f"{config.FIGURES_DIR}/exp3_range_by_tier_heatmap.png",
        cbar_label="Range",
        cmap="YlGnBu",
    )


def _line_and_heatmap(tier_df, value_col, line_ylabel, line_title, line_out,
                       heat_title, heat_out, cbar_label, cmap="YlOrRd"):
    agg = tier_df.groupby(["tier", "instrument"])[value_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    for inst, grp in agg.groupby("instrument"):
        ax.plot(grp["tier"], grp[value_col], marker="o", label=inst)
    ax.set_xlabel("Coverage Tier (1=Covered -> 4=Absent Latin)")
    ax.set_ylabel(line_ylabel)
    ax.set_title(line_title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(line_out, bbox_inches="tight")
    plt.close(fig)


    matrix = agg.pivot_table(index="tier", columns="instrument", values=value_col)
    pu.annotated_heatmap(
        matrix,
        title=heat_title,
        out_path=heat_out,
        cmap=cmap, fmt="{:.2f}", cbar_label=cbar_label,
        ylabel="Coverage Tier",
    )
