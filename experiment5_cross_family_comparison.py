"""
experiment5_cross_family_comparison.py — Experiment 5 & Decisive Tests
─────────────────────────────────────────────────────────────────────────────
Executes the plan's 4 family comparisons at system level (segment level
noted below), incorporating the decisive Williams test and 10,000-permutation
paired tests for P-values.

"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import config
import stats_utils as su
import plot_utils as pu
from data_loader import load_all_scores, oriented, wide_by_instrument

logger = logging.getLogger(__name__)


def run():
    logger.info(f"Executing Experiment 5: Cross-Family Comparison & Decisive Tests — WMT{config.YEAR}...")

    if not config.LLM_JUDGE_INSTRUMENTS and not config.HUMAN_INSTRUMENTS:
        logger.info("No LLM-judge or human instruments configured yet — 'metrics vs judges', "
                     "'metrics vs human', and 'judges vs human' comparisons will produce empty rows "
                     "until data/wmt%s_judge_scores.csv / human_scores.csv are populated and "
                     "config.LLM_JUDGE_INSTRUMENTS / HUMAN_INSTRUMENTS are filled in." % config.YEAR)

    seg_df = load_all_scores(level="segment")
    if seg_df.empty:
        logger.info(f"No segment-level data for WMT{config.YEAR} — running system-level only "
                     f"(plan calls for 'segment where available'; none is available yet).")

    df_sys = oriented(load_all_scores(level="system"))
    manipuri_pairs = [p for p in config.LANGUAGE_PAIRS if "Manipuri" in p]
    comp_rows = []

    # 1. Metrics vs Metrics (System level, all pairs)
    _eval_family_pair(df_sys, config.METRIC_INSTRUMENTS, config.METRIC_INSTRUMENTS, config.LANGUAGE_PAIRS, "metrics_vs_metrics", comp_rows)

    # 2. Metrics vs LLM judges (System level, all pairs)
    _eval_family_pair(df_sys, config.METRIC_INSTRUMENTS, config.LLM_JUDGE_INSTRUMENTS, config.LANGUAGE_PAIRS, "metrics_vs_judges", comp_rows)

    # 3 & 4. Manipuri pairs only (system level; segment level once available)
    _eval_family_pair(df_sys, config.METRIC_INSTRUMENTS, config.HUMAN_INSTRUMENTS, manipuri_pairs, "metrics_vs_human_system", comp_rows)
    _eval_family_pair(df_sys, config.LLM_JUDGE_INSTRUMENTS, config.HUMAN_INSTRUMENTS, manipuri_pairs, "judges_vs_human_system", comp_rows)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(f"{config.TABLES_DIR}/exp5_cross_family_correlations.csv", index=False)
    _plot_metrics_vs_metrics_heatmap(comp_df)

    # DECISIVE RESULT: Williams Test & Paired Permutation Test
    williams_rows = []
    for pair in manipuri_pairs:
        wide = wide_by_instrument(df_sys, pair)
        if "Human_DA" not in wide.columns:
            continue

        best_metric, max_r = None, -1.0
        for m in config.METRIC_INSTRUMENTS:
            if m in wide.columns:
                r_val = su.pearson_r(wide[m].values, wide["Human_DA"].values)
                if not np.isnan(r_val) and r_val > max_r:
                    max_r, best_metric = r_val, m

        if not best_metric:
            continue

        best_metric_human_agreements = su.agreement_vector(wide[best_metric].values, wide["Human_DA"].values)

        for judge in config.LLM_JUDGE_INSTRUMENTS:
            if judge not in wide.columns:
                continue

            r_jh = su.pearson_r(wide[judge].values, wide["Human_DA"].values)
            r_mh = max_r
            r_jm = su.pearson_r(wide[judge].values, wide[best_metric].values)

            wt = su.williams_test(r_jh, r_mh, r_jm, n=len(wide))

            judge_human_agreements = su.agreement_vector(wide[judge].values, wide["Human_DA"].values)
            perm_test = su.paired_permutation_test(
                judge_human_agreements, best_metric_human_agreements,
                n_permutations=config.PERMUTATION_N
            )

            williams_rows.append({
                "language_pair": pair,
                "judge": judge,
                "best_metric": best_metric,
                "r_judge_human": r_jh,
                "r_metric_human": r_mh,
                "r_judge_metric": r_jm,
                "williams_t": wt["t"],
                "williams_p": wt["p"],
                "P_judge_human": np.mean(judge_human_agreements),
                "P_metric_human": np.mean(best_metric_human_agreements),
                "P_diff_perm_p_val": perm_test["p_value"],
            })

    williams_df = pd.DataFrame(williams_rows)
    if not williams_df.empty:
        w_adjs = su.holm_bonferroni(williams_df["williams_p"].fillna(1.0).values)
        williams_df["williams_p_holm_adj"] = [p["p_adj"] for p in w_adjs]
        williams_df["williams_reject_H0"] = [p["reject"] for p in w_adjs]

        p_adjs = su.holm_bonferroni(williams_df["P_diff_perm_p_val"].fillna(1.0).values)
        williams_df["P_diff_p_holm_adj"] = [p["p_adj"] for p in p_adjs]
        williams_df["P_diff_reject_H0"] = [p["reject"] for p in p_adjs]
    else:
        logger.info("No Williams/decisive-test rows produced — requires Human_DA and at least one "
                     "LLM-judge instrument to be populated for a Manipuri pair.")

    williams_df.to_csv(f"{config.TABLES_DIR}/exp5_decisive_tests_results.csv", index=False)

    _plot_human_correlations(comp_df)
    _plot_decisive_tests(williams_df)

    print("\nExperiment 5: Decisive Tests Results:")
    print(williams_df.to_string(index=False) if not williams_df.empty else "  (no rows — see log above)")


def _plot_metrics_vs_metrics_heatmap(comp_df: pd.DataFrame):
    """Instrument x instrument Tau-b for the 'metrics vs metrics' comparison
    — internal concordance of the automatic-metric family, Deliverable 1's
    'Table (cross-family tau, rho, P...)' rendered as a heatmap."""
    sub = comp_df[comp_df["comparison"] == "metrics_vs_metrics"]
    if sub.empty:
        return
    insts = sorted(set(sub["instrument_a"]) | set(sub["instrument_b"]))
    matrix = pd.DataFrame(index=insts, columns=insts, dtype=float)
    for _, r in sub.iterrows():
        matrix.loc[r["instrument_a"], r["instrument_b"]] = r["mean_tau_b"]
        matrix.loc[r["instrument_b"], r["instrument_a"]] = r["mean_tau_b"]
    for i in insts:
        matrix.loc[i, i] = 1.0
    pu.annotated_heatmap(
        matrix,
        title=f"Experiment 5: Metrics-vs-Metrics Tau-b Concordance — WMT{config.YEAR}",
        out_path=f"{config.FIGURES_DIR}/exp5_metrics_vs_metrics_heatmap.png",
        cmap="coolwarm", vmin=-1, vmax=1, fmt="{:.2f}", cbar_label="Kendall's Tau-b",
    )


def _plot_decisive_tests(williams_df: pd.DataFrame):

    if williams_df.empty:
        return
    label = williams_df["language_pair"].astype(str) + " / " + williams_df["judge"].astype(str)
    plot_df = pd.DataFrame({
        "comparison": label,
        "r_judge_human": williams_df["r_judge_human"],
        "r_metric_human": williams_df["r_metric_human"],
    }).melt(id_vars="comparison", var_name="which", value_name="r")

    pu.grouped_bar(
        plot_df, x="comparison", y="r", group="which",
        title=f"Experiment 5: Judge vs Best-Metric Correlation with Human — WMT{config.YEAR}",
        ylabel="Pearson r with Human_DA",
        out_path=f"{config.FIGURES_DIR}/exp5_decisive_tests_r_comparison.png",
        figsize=(max(9, 0.7 * len(label) + 3), 5.5),
    )

    pu.grouped_bar(
        williams_df.assign(comparison=label), x="comparison", y="P_diff_p_holm_adj",
        title=f"Experiment 5: Holm-Adjusted P-Difference Permutation p-values — WMT{config.YEAR}",
        ylabel="Holm-adjusted p-value",
        out_path=f"{config.FIGURES_DIR}/exp5_decisive_tests_p_values.png",
        figsize=(max(9, 0.7 * len(label) + 3), 5.5),
    )


def _eval_family_pair(df, fam_a, fam_b, pairs, label, rows):
    same_family = list(fam_a) == list(fam_b)
    combos = itertools.combinations(fam_a, 2) if same_family else itertools.product(fam_a, fam_b)

    for a, b in combos:
        if a == b:
            continue
        taus, ps, rhos = [], [], []
        for pair in pairs:
            wide = wide_by_instrument(df, pair)
            if a in wide.columns and b in wide.columns:
                sub = wide[[a, b]].dropna()
                if len(sub) >= 3:
                    taus.append(su.kendall_tau_b(sub[a].values, sub[b].values))
                    ps.append(su.pairwise_P(sub[a].values, sub[b].values)["P"])
                    rhos.append(su.spearman_rho(sub[a].values, sub[b].values))

        if taus:
            pt_t, lo_t, hi_t = su.bootstrap_ci(taus, np.mean)
            pt_p, lo_p, hi_p = su.bootstrap_ci(ps, np.mean)
            pt_r, lo_r, hi_r = su.bootstrap_ci(rhos, np.mean)
            rows.append({
                "comparison": label,
                "instrument_a": a,
                "instrument_b": b,
                "mean_tau_b": pt_t,
                "tau_ci_low": lo_t,
                "tau_ci_high": hi_t,
                "mean_P": pt_p,
                "P_ci_low": lo_p,
                "P_ci_high": hi_p,
                "mean_rho": pt_r,
                "rho_ci_low": lo_r,
                "rho_ci_high": hi_r,
            })


def _plot_human_correlations(comp_df):
    if comp_df.empty:
        return

    human_comps = comp_df[comp_df['comparison'].isin(['metrics_vs_human_system', 'judges_vs_human_system'])]
    if human_comps.empty:
        return

    plot_df = human_comps[human_comps['instrument_b'].isin(config.HUMAN_INSTRUMENTS)].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values('mean_tau_b', ascending=False)

    plt.figure(figsize=(10, 6))

    lower_error = np.maximum(0, plot_df['mean_tau_b'] - plot_df['tau_ci_low'])
    upper_error = np.maximum(0, plot_df['tau_ci_high'] - plot_df['mean_tau_b'])
    errors = [lower_error.fillna(0).values, upper_error.fillna(0).values]

    colors = ['mediumpurple' if comp == 'metrics_vs_human_system' else 'darkorange' for comp in plot_df['comparison']]

    plt.bar(plot_df['instrument_a'], plot_df['mean_tau_b'], yerr=errors, capsize=5, color=colors, edgecolor='black')

    plt.title(f'Experiment 5: Correlation with Human Baseline (Kendall\'s Tau-b) — WMT{config.YEAR}', fontweight='bold')
    plt.ylabel('Mean Tau-b against Human')
    plt.xlabel('Instrument (Purple = Metric, Orange = Judge)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}/exp5_human_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()
