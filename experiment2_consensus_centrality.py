"""
experiment2_consensus_centrality.py — Experiment 2: Consensus Centrality
─────────────────────────────────────────────────────────────────────────────
Computes Types II & IV inter-instrument correlations, Kendall's W, leave-one-out
centrality, tau-b heatmaps, and 10,000-permutation tests for P values.

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
    logger.info(f"Executing Experiment 2: Consensus Centrality (Types II & IV) — WMT{config.YEAR}...")
    df = oriented(load_all_scores(level="system"))

    scopes = {
        "metrics": config.METRIC_INSTRUMENTS,
        "metrics_plus_judges": config.METRIC_INSTRUMENTS + config.LLM_JUDGE_INSTRUMENTS,
        "all_incl_human": config.METRIC_INSTRUMENTS + config.LLM_JUDGE_INSTRUMENTS + config.HUMAN_INSTRUMENTS,
    }

    run_scopes = {}
    seen_signatures = {}
    for scope_name, inst_list in scopes.items():
        sig = tuple(sorted(inst_list))
        if sig in seen_signatures:
            logger.info(f"Scope '{scope_name}' is identical to '{seen_signatures[sig]}' "
                        f"(no judge/human data loaded yet) — skipping duplicate computation.")
            continue
        seen_signatures[sig] = scope_name
        run_scopes[scope_name] = inst_list

    for scope_name, inst_list in run_scopes.items():
        qc_rows = []
        loo_accumulator = {inst: {"tau": [], "P": [], "ties": [], "agreements": []} for inst in inst_list}

        global_agreements = {a: {b: {} for b in inst_list if b != a} for a in inst_list}

        for pair in config.LANGUAGE_PAIRS:
            wide = wide_by_instrument(df, pair)
            present = [i for i in inst_list if i in wide.columns]
            if len(present) < 2:
                continue

            mat = wide[present].dropna().values.T
            w_val = su.kendalls_w(mat) if mat.shape[1] >= 2 else np.nan

            taus, ps, tie_rates = [], [], []
            for a, b in itertools.combinations(present, 2):
                x, y = wide[a].values, wide[b].values
                p_info = su.pairwise_P(x, y)
                t_val = su.kendall_tau_b(x, y)

                if not np.isnan(p_info["P"]):
                    ps.append(p_info["P"])
                    tie_rates.append(p_info["tie_rate"])
                    agr_vec = su.agreement_vector(x, y)
                    global_agreements[a][b][pair] = agr_vec
                    global_agreements[b][a][pair] = agr_vec

                if not np.isnan(t_val):
                    taus.append(t_val)

            qc_rows.append({
                "language_pair": pair,
                "kendalls_W": w_val,
                "mean_P": np.mean(ps) if ps else np.nan,
                "mean_tie_rate": np.mean(tie_rates) if tie_rates else np.nan,
                "mean_tau_b": np.mean(taus) if taus else np.nan,
            })

            scores_dict = {i: wide[i].values for i in present}
            loo = su.leave_one_out_centrality(scores_dict)
            for inst, metrics in loo.items():
                if not np.isnan(metrics["mean_tau"]):
                    loo_accumulator[inst]["tau"].append(metrics["mean_tau"])
                    loo_accumulator[inst]["P"].append(metrics["mean_P"])
                    loo_accumulator[inst]["ties"].append(metrics["mean_tie_rate"])

        qc_df = pd.DataFrame(qc_rows)
        qc_df.to_csv(f"{config.TABLES_DIR}/exp2_qc_level_W_and_P_{scope_name}.csv", index=False)
        _plot_qc_level(qc_df, scope_name)

        loo_rows = []
        for inst in inst_list:
            t_vals = loo_accumulator[inst]["tau"]
            p_vals = loo_accumulator[inst]["P"]
            tie_vals = loo_accumulator[inst]["ties"]
            loo_rows.append({
                "instrument": inst,
                "study_mean_tau": np.mean(t_vals) if t_vals else np.nan,
                "study_mean_P": np.mean(p_vals) if p_vals else np.nan,
                "mean_tie_rate": np.mean(tie_vals) if tie_vals else np.nan,
            })

        loo_df = pd.DataFrame(loo_rows).sort_values("study_mean_tau", ascending=False)

        if not loo_df.empty:
            top_inst = loo_df.iloc[0]["instrument"]
            p_val_perms = []
            for inst in inst_list:
                if inst == top_inst:
                    p_val_perms.append(1.0)
                    continue

                top_agreements, inst_agreements = [], []
                for other in inst_list:
                    if other != top_inst and other != inst:
                        top_by_pair = global_agreements[top_inst].get(other, {})
                        inst_by_pair = global_agreements[inst].get(other, {})

                        common_pairs = set(top_by_pair) & set(inst_by_pair)
                        for pr in sorted(common_pairs):
                            top_agreements.extend(top_by_pair[pr])
                            inst_agreements.extend(inst_by_pair[pr])

                if top_agreements and inst_agreements:
                    perm_res = su.paired_permutation_test(
                        np.array(top_agreements), np.array(inst_agreements),
                        n_permutations=config.PERMUTATION_N
                    )
                    p_val_perms.append(perm_res["p_value"])
                else:
                    p_val_perms.append(np.nan)

            perm_dict = dict(zip(inst_list, p_val_perms))
            loo_df["perm_p_val_vs_top"] = loo_df["instrument"].map(perm_dict)

            others_df = loo_df[loo_df["instrument"] != top_inst].dropna(subset=["perm_p_val_vs_top"])
            valid_p = others_df["perm_p_val_vs_top"].values

            adj_dict = {top_inst: 1.0}
            if len(valid_p) > 0:
                adjusted = su.holm_bonferroni(valid_p)
                for k, v in zip(others_df["instrument"], adjusted):
                    adj_dict[k] = v["p_adj"]

            loo_df["perm_p_adj_vs_top"] = loo_df["instrument"].map(adj_dict)

        loo_df.to_csv(f"{config.TABLES_DIR}/exp2_leave_one_out_centrality_{scope_name}.csv", index=False)
        _plot_heatmap(df, inst_list, scope_name)
        _plot_loo_centrality(loo_df, scope_name)

    print("\nExperiment 2: Consensus Centrality Completed.")


def _plot_heatmap(df, inst_list, scope_name):
    if len(inst_list) < 2:
        return
    matrix = pd.DataFrame(index=inst_list, columns=inst_list, dtype=float)
    for a, b in itertools.product(inst_list, inst_list):
        if a == b:
            matrix.loc[a, b] = 1.0
            continue
        taus = []
        for pair in config.LANGUAGE_PAIRS:
            wide = wide_by_instrument(df, pair)
            if a in wide.columns and b in wide.columns:
                sub = wide[[a, b]].dropna()
                t = su.kendall_tau_b(sub[a].values, sub[b].values)
                if not np.isnan(t):
                    taus.append(t)
        matrix.loc[a, b] = np.mean(taus) if taus else np.nan

    pu.annotated_heatmap(
        matrix,
        title=f"Tau-b Heatmap ({scope_name}) — WMT{config.YEAR}",
        out_path=f"{config.FIGURES_DIR}/exp2_instrument_heatmap_{scope_name}.png",
        cmap="coolwarm", vmin=-1, vmax=1, fmt="{:.2f}", cbar_label="Kendall's Tau-b",
    )


def _plot_qc_level(qc_df: pd.DataFrame, scope_name: str):
    """Per-language-pair Kendall's W and mean P — the plan's Deliverable
    'Table (QC-level W and P per language pair)', plotted directly."""
    if qc_df.empty:
        return
    plot_df = qc_df.melt(id_vars="language_pair", value_vars=["kendalls_W", "mean_P"],
                          var_name="statistic", value_name="value")
    pu.grouped_bar(
        plot_df, x="language_pair", y="value", group="statistic",
        title=f"Experiment 2: Kendall's W and Mean P by Language Pair ({scope_name}) — WMT{config.YEAR}",
        ylabel="Value (0-1)",
        out_path=f"{config.FIGURES_DIR}/exp2_qc_level_W_and_P_{scope_name}.png",
        figsize=(max(9, 0.5 * qc_df['language_pair'].nunique() + 3), 5.5),
    )


def _plot_loo_centrality(loo_df, scope_name):
    if loo_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(loo_df['instrument'], loo_df['study_mean_tau'], color='mediumseagreen', edgecolor='black')
    axes[0].set_title("Mean Kendall's Tau-b\nagainst all others")
    axes[0].set_xticks(range(len(loo_df)))
    axes[0].set_xticklabels(loo_df['instrument'], rotation=45, ha='right')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    axes[1].bar(loo_df['instrument'], loo_df['study_mean_P'], color='cornflowerblue', edgecolor='black')
    axes[1].set_title("Mean Pairwise Agreement P\nagainst all others")
    axes[1].set_xticks(range(len(loo_df)))
    axes[1].set_xticklabels(loo_df['instrument'], rotation=45, ha='right')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle(f'Experiment 2: Leave-One-Out Centrality ({scope_name}) — WMT{config.YEAR}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}/exp2_loo_centrality_{scope_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
