"""
experiment1_discriminative_power.py — Experiment 1 & Narrative (b) System Analysis
─────────────────────────────────────────────────────────────────────────────
Computes Type I Discriminative Power (CV*, range, effective distinct values)
at three levels — QC (per language pair), Category, and Study — plus
Narrative (b)'s circularity breakdown by system backbone.

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


def _infer_backbone(system_name):

    sys_lower = str(system_name).lower()
    if "indictrans2" in sys_lower or "indic trans2" in sys_lower:
        return "indictrans2"
    elif "nllb" in sys_lower:
        return "nllb"
    elif "sarvam" in sys_lower:
        return "sarvam"
    else:
        return "scratch_transformer"


def run(instruments=None, include_judges=True):
    logger.info(f"Executing Experiment 1: Discriminative Power (Type I) — WMT{config.YEAR}...")
    df = load_all_scores(level="system")

    if "backbone" not in df.columns:
        logger.info("Column 'backbone' missing — inferring from system names (heuristic, see docstring).")
        df["backbone"] = df["system"].apply(_infer_backbone)

    inst_list = instruments or (config.METRIC_INSTRUMENTS + (config.LLM_JUDGE_INSTRUMENTS if include_judges else []))

    # ─── Level 1: QC (per language pair) ────────────────────────────────────
    qc_rows = []
    for pair in config.LANGUAGE_PAIRS:
        wide = wide_by_instrument(df, pair)
        if wide.empty:
            continue

        category = wide["category"].iloc[0] if "category" in wide.columns else None

        for inst in inst_list:
            if inst not in wide.columns:
                continue
            vals = wide[inst].dropna().values
            if len(vals) < 2:
                continue

            qc_rows.append({
                "language_pair": pair,
                "category": category,
                "direction": "en2indic" if pair.startswith("English") else "indic2en",
                "instrument": inst,
                "n_systems": len(vals),
                "cv_star": su.cv_star(vals),
                "range": su.observed_range(vals),
                "effective_n_distinct": su.effective_num_distinct_values(vals),
            })

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(f"{config.TABLES_DIR}/exp1_cv_star_qc_level.csv", index=False)
    _plot_qc_heatmap(qc_df)

    # ─── Level 2: Category ──────────────────────────────────────────────────
    has_categories = "category" in qc_df.columns and qc_df["category"].notna().any() and (qc_df["category"] != "").any()
    if has_categories:
        cat_rows = []
        for (cat, inst), group in qc_df[qc_df["category"].notna() & (qc_df["category"] != "")].groupby(["category", "instrument"]):
            cvs = group["cv_star"].dropna().values
            pt, lo, hi = su.bootstrap_ci(cvs, np.mean, n_resamples=config.BOOTSTRAP_B)
            cat_rows.append({
                "category": cat,
                "instrument": inst,
                "cv_star_category_mean": pt,
                "cv_star_ci_low": lo,
                "cv_star_ci_high": hi,
                "range_mean": group["range"].mean(),
                "n_language_pairs": group["language_pair"].nunique(),
            })
        cat_df = pd.DataFrame(cat_rows).sort_values(["category", "cv_star_category_mean"], ascending=[True, False])
        cat_df.to_csv(f"{config.TABLES_DIR}/exp1_cv_star_category_level.csv", index=False)
        _plot_category_level(cat_df)
    else:
        logger.info(f"No Category tags in the WMT{config.YEAR} data — skipping category-level table "
                    f"(this is expected for WMT25, which doesn't tag Category 1/2).")
        cat_df = pd.DataFrame(columns=["category", "instrument", "cv_star_category_mean",
                                        "cv_star_ci_low", "cv_star_ci_high", "range_mean",
                                        "n_language_pairs"])
        cat_df.to_csv(f"{config.TABLES_DIR}/exp1_cv_star_category_level.csv", index=False)

    # ─── Level 3: Study ──────────────────────────────────────────────────────
    study_rows = []
    for inst, group in qc_df.groupby("instrument"):
        cvs = group["cv_star"].dropna().values
        pt, lo, hi = su.bootstrap_ci(cvs, np.mean, n_resamples=config.BOOTSTRAP_B)
        study_rows.append({
            "instrument": inst,
            "cv_star_study_mean": pt,
            "cv_star_ci_low": lo,
            "cv_star_ci_high": hi,
            "range_mean": group["range"].mean(),
        })

    study_df = pd.DataFrame(study_rows).sort_values("cv_star_study_mean", ascending=False)
    study_df.to_csv(f"{config.TABLES_DIR}/exp1_cv_star_study_level.csv", index=False)

    # ─── Narrative (b): system-type / circularity breakdown ────────────────
    narrative_b_df = _narrative_b_analysis(df, inst_list)
    narrative_b_df.to_csv(f"{config.TABLES_DIR}/narrative_b_system_type_analysis.csv", index=False)
    _plot_narrative_b(narrative_b_df)

    _plot_cv_star(study_df)

    print("\nExperiment 1: Discriminative Power Study Level Summary:")
    print(study_df.to_string(index=False))
    return {"qc_level": qc_df, "category_level": cat_df, "study_level": study_df,
            "narrative_b": narrative_b_df}


def _narrative_b_analysis(df: pd.DataFrame, inst_list) -> pd.DataFrame:

    df_o = oriented(df)
    per_system_rows = []

    for pair in config.LANGUAGE_PAIRS:
        wide = wide_by_instrument(df_o, pair)
        if wide.empty or "backbone" not in wide.columns:
            continue
        present = [i for i in inst_list if i in wide.columns]
        if len(present) < 2:
            continue

        ranks = wide[present].rank(ascending=False, method="average")  # 1 = best

        for row_idx in wide.index:

            rank_vals = ranks.loc[row_idx, present].astype(float).values
            per_system_rows.append({
                "language_pair": pair,
                "team": wide.loc[row_idx, "team"],
                "backbone": wide.loc[row_idx, "backbone"],
                "mean_rank_this_pair": float(np.mean(rank_vals)),
                "cv_star_own_standing": su.cv_star(rank_vals),
            })

    per_system_df = pd.DataFrame(per_system_rows)
    if per_system_df.empty:
        logger.warning("Narrative (b): no rows produced (need >=2 instruments present per pair).")
        return pd.DataFrame(columns=["backbone", "category_label", "n_primary_systems",
                                      "mean_rank_across_instruments", "cv_star_own_standing_mean",
                                      "P_within_type_neural_vs_surface"])

    agg = per_system_df.groupby("backbone").agg(
        n_primary_systems=("team", "count"),
        mean_rank_across_instruments=("mean_rank_this_pair", "mean"),
        cv_star_own_standing_mean=("cv_star_own_standing", "mean"),
    ).reset_index()

    # P restricted to within-type system pairs: neural-vs-surface agreement,
    # computed only among same-backbone systems within each language pair.
    neural, surface = config.NARRATIVE_B_NEURAL_METRIC, config.NARRATIVE_B_SURFACE_METRIC
    p_by_backbone = {bb: [] for bb in agg["backbone"]}
    for pair in config.LANGUAGE_PAIRS:
        wide = wide_by_instrument(df_o, pair)
        if wide.empty or "backbone" not in wide.columns:
            continue
        if neural not in wide.columns or surface not in wide.columns:
            continue
        for bb, grp in wide.groupby("backbone"):
            if bb not in p_by_backbone or len(grp) < 2:
                continue
            p_info = su.pairwise_P(grp[neural].values, grp[surface].values)
            if not np.isnan(p_info["P"]):
                p_by_backbone[bb].append(p_info["P"])

    agg["P_within_type_neural_vs_surface"] = agg["backbone"].map(
        lambda bb: float(np.mean(p_by_backbone[bb])) if p_by_backbone.get(bb) else np.nan
    )
    agg["category_label"] = agg["backbone"].map(lambda bb: config.SYSTEM_BACKBONE_CATEGORY.get(bb, "Other"))
    agg.attrs["neural_metric"] = neural
    agg.attrs["surface_metric"] = surface

    return agg.sort_values("mean_rank_across_instruments")[
        ["backbone", "category_label", "n_primary_systems", "mean_rank_across_instruments",
         "cv_star_own_standing_mean", "P_within_type_neural_vs_surface"]
    ]


def _plot_cv_star(study_df):
    """Plots CV* Study Mean with Bootstrap CIs."""
    if study_df.empty:
        return
    pu.grouped_bar(
        study_df, x="instrument", y="cv_star_study_mean",
        yerr_low="cv_star_ci_low", yerr_high="cv_star_ci_high",
        title=f"Experiment 1: Discriminative Power (CV*) by Instrument — WMT{config.YEAR}",
        ylabel="Study Mean CV* (%)",
        out_path=f"{config.FIGURES_DIR}/exp1_cv_star_comparison.png",
    )


def _plot_qc_heatmap(qc_df: pd.DataFrame):
    """CV* per (language pair x instrument) — one glance at every QC-level
    cell the plan's Deliverable 1 table contains, numbers visible in-cell."""
    if qc_df.empty:
        return
    matrix = qc_df.pivot_table(index="language_pair", columns="instrument", values="cv_star", aggfunc="first")
    matrix = matrix.reindex([p for p in config.LANGUAGE_PAIRS if p in matrix.index])
    pu.annotated_heatmap(
        matrix,
        title=f"Experiment 1: CV* by Language Pair x Instrument (QC level) — WMT{config.YEAR}",
        out_path=f"{config.FIGURES_DIR}/exp1_cv_star_qc_heatmap.png",
        cmap="YlOrRd", fmt="{:.0f}", cbar_label="CV* (%)",
    )


def _plot_category_level(cat_df: pd.DataFrame):
    if cat_df.empty:
        return
    pu.grouped_bar(
        cat_df, x="instrument", y="cv_star_category_mean", group="category",
        yerr_low="cv_star_ci_low", yerr_high="cv_star_ci_high",
        title=f"Experiment 1: CV* by Category x Instrument — WMT{config.YEAR}",
        ylabel="Category Mean CV* (%)",
        out_path=f"{config.FIGURES_DIR}/exp1_cv_star_category_level.png",
    )


def _plot_narrative_b(nb_df: pd.DataFrame):
    """Three-panel bar chart: mean rank, CV* of own standing, and within-type
    P — the three quantities the plan's Narrative (b) asks for, per backbone."""
    if nb_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = nb_df["category_label"].astype(str)
    x = np.arange(len(labels))

    axes[0].bar(x, nb_df["mean_rank_across_instruments"], color="steelblue", edgecolor="black")
    axes[0].set_title("Mean Rank Across Instruments\n(1 = best)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].bar(x, nb_df["cv_star_own_standing_mean"], color="darkorange", edgecolor="black")
    axes[1].set_title("CV* of Own Standing\nAcross Instruments (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    p_vals = nb_df["P_within_type_neural_vs_surface"]
    axes[2].bar(x, p_vals.fillna(0), color=["seagreen" if pd.notna(v) else "lightgray" for v in p_vals], edgecolor="black")
    axes[2].set_title(f"Within-Type P\n({config.NARRATIVE_B_NEURAL_METRIC} vs {config.NARRATIVE_B_SURFACE_METRIC})")
    axes[2].set_ylim(0, 1)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"Narrative (b): Circularity Breakdown by System Backbone — WMT{config.YEAR}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}/narrative_b_system_type_analysis.png", dpi=300)
    plt.close(fig)
