"""
main.py — Main Pipeline Orchestrator & Narrative Synthesis
─────────────────────────────────────────────────────────────────────────────
Runs Experiments 1 through 5 for the selected WMT year, then synthesises
Narrative (a) and Narrative (b) reports FROM THE COMPUTED TABLES.

"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd


def _read_csv_or_none(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df if not df.empty else None
        except Exception:
            return None
    return None


def generate_narrative_reports(config):

    tdir = config.TABLES_DIR
    lines_a = [
        "=" * 79,
        f"NARRATIVE (a): Metric Trustworthiness & Script Coverage — WMT{config.YEAR} Low-Resource Indic",
        "=" * 79,
        "",
        "Synthesised from Experiments 1, 3, 4 and the metric-vs-human row of Experiment 5.",
        "Numbers below are read directly from outputs/wmt%d/tables/. A section marked" % config.YEAR,
        "'not available' means that leg of evidence doesn't exist for this year's data",
        "",
    ]

    exp1 = _read_csv_or_none(f"{tdir}/exp1_cv_star_study_level.csv")
    if exp1 is not None:
        lines_a.append("1. Discriminative power (CV*, study level; higher = instrument spreads systems out more):")
        for _, r in exp1.sort_values("cv_star_study_mean", ascending=False).iterrows():
            lines_a.append(f"   - {r['instrument']}: CV*={r['cv_star_study_mean']:.1f}% "
                            f"[{r['cv_star_ci_low']:.1f}, {r['cv_star_ci_high']:.1f}], "
                            f"mean observed range={r['range_mean']:.2f}")
        lines_a.append("")
    else:
        lines_a.append("1. Discriminative power: not available (run Experiment 1 first).\n")

    exp3_tier = _read_csv_or_none(f"{tdir}/exp3_inflation_by_tier.csv")
    if exp3_tier is not None:
        lines_a.append("2. Range-normalised (top - median) gap by coverage tier (1=covered script/language -> 4=absent-Latin):")
        agg = exp3_tier.groupby(["tier", "instrument"])["range_normalised_gap"].mean().reset_index()
        for tier, grp in agg.groupby("tier"):
            top = grp.sort_values("range_normalised_gap", ascending=False).iloc[0]
            lines_a.append(f"   - Tier {int(tier)}: largest gap = {top['instrument']} "
                            f"({top['range_normalised_gap']:.3f})")
        lines_a.append("")
    else:
        lines_a.append("2. Coverage-tier inflation: not available (no tier-mapped rows for this year's data).\n")

    exp3_shift = _read_csv_or_none(f"{tdir}/exp3_paired_script_shifts.csv")
    if exp3_shift is not None:
        lines_a.append("   Paired Meitei-Mayek minus Bengali script shift (same teams/content, script only differs):")
        for _, r in exp3_shift.iterrows():
            lines_a.append(f"   - {r['direction']} / {r['instrument']}: mean shift="
                            f"{r['mean_shift_meitei_minus_bengali']:.2f} "
                            f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}], n={int(r['n_paired_systems'])}")
        lines_a.append("")
    else:
        lines_a.append(f"   Paired script-shift table: not available for WMT{config.YEAR} "
                        f"(needs dual-script Manipuri submissions — present only in WMT26).\n")

    exp4 = _read_csv_or_none(f"{tdir}/exp4_floors_and_inflation_ratios.csv")
    if exp4 is not None:
        lines_a.append("3. Null-output floor-inflation ratio (degenerate-system score / pair's top-system score):")
        for _, r in exp4.sort_values("mean_floor_inflation_ratio", ascending=False).iterrows():
            lines_a.append(f"   - {r['instrument']}: mean floor={r['mean_floor']:.2f}, "
                            f"mean ratio={r['mean_floor_inflation_ratio']:.3f}, "
                            f"max ratio={r['max_floor_inflation_ratio']:.3f}")
        lines_a.append("")
    else:
        lines_a.append(f"3. Null-output validity: no primary systems with BLEU <= "
                        f"{config.DEGENERATE_BLEU_MAX} found for WMT{config.YEAR}.\n")

    exp5 = _read_csv_or_none(f"{tdir}/exp5_cross_family_correlations.csv")
    human_rows = exp5[exp5["comparison"] == "metrics_vs_human_system"] if exp5 is not None else None
    if human_rows is not None and not human_rows.empty:
        lines_a.append("4. Metric-vs-human validity (Manipuri only):")
        for _, r in human_rows.sort_values("mean_tau_b", ascending=False).iterrows():
            lines_a.append(f"   - {r['instrument_a']} vs {r['instrument_b']}: tau_b={r['mean_tau_b']:.2f}")
        lines_a.append("")
    else:
        lines_a.append(f"4. Metric-vs-human validity: not available yet — populate "
                        f"{os.path.basename(config.HUMAN_SCORES_CSV)} and config.HUMAN_INSTRUMENTS "
                        f"to enable this leg.\n")

    lines_a.append("Per RQ1, any recommendation must stay TIER-CONDITIONAL, not blanket: read the")
    lines_a.append("Tier 1-4 gap numbers (Experiment 3) alongside the CV*/floor numbers (Experiments")
    lines_a.append("1 and 4) above before endorsing any single instrument for a given language.")
    lines_a.append("=" * 79)

    with open(os.path.join(config.OUTPUT_DIR, "narrative_a_summary.txt"), "w") as f:
        f.write("\n".join(lines_a))

    # ─── Narrative (b) ───────────────────────────────────────────────────
    lines_b = [
        "=" * 79,
        f"NARRATIVE (b): System Standing & Circularity Hypothesis (RQ4) — WMT{config.YEAR}",
        "=" * 79,
        "",
    ]
    nb = _read_csv_or_none(f"{tdir}/narrative_b_system_type_analysis.csv")
    if nb is not None:
        neural, surface = config.NARRATIVE_B_NEURAL_METRIC, config.NARRATIVE_B_SURFACE_METRIC
        lines_b.append(f"Per system-type (backbone) breakdown, primary submissions only "
                        f"(within-type P uses {neural} vs {surface} as the neural-vs-surface probe):")
        for _, r in nb.iterrows():
            p_val = r.get("P_within_type_neural_vs_surface", np.nan)
            p_str = f"{p_val:.2f}" if pd.notna(p_val) else "n/a (fewer than 2 same-backbone systems in any pair)"
            lines_b.append(
                f"   - {r['category_label']} (n={int(r['n_primary_systems'])} primary systems): "
                f"mean rank across instruments={r['mean_rank_across_instruments']:.2f} (1=best), "
                f"CV* of own standing across instruments={r['cv_star_own_standing_mean']:.1f}%, "
                f"within-type P={p_str}"
            )
        lines_b.append("")
        lines_b.append("Circularity read (RQ4): a backbone whose within-type P or CV*-of-own-standing")
        lines_b.append("stands out from the others is the signal this test is looking for — treat it")
        lines_b.append("as a lead to investigate given the small per-type sample sizes here, not as")
        lines_b.append("a proven bias. Backbone labels are inferred heuristically from system names")
        lines_b.append("(see experiment1_discriminative_power._infer_backbone) and should be checked")
        lines_b.append("against teams' system-description papers once available.")
    else:
        lines_b.append("narrative_b_system_type_analysis.csv not found or empty — run Experiment 1 first.")

    lines_b.append("=" * 79)
    with open(os.path.join(config.OUTPUT_DIR, "narrative_b_summary.txt"), "w") as f:
        f.write("\n".join(lines_b))


def main():
    parser = argparse.ArgumentParser(description="QRA Meta-Evaluation Pipeline")
    parser.add_argument("--year", type=int, choices=[2025, 2026], default=None,
                         help="WMT Indic-MT dataset year to run against (overrides WMT_YEAR env var).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.year is not None:
        os.environ["WMT_YEAR"] = str(args.year)

    # config.py reads WMT_YEAR at import time, so import it (and everything
    # that transitively imports it) only after the env var is fixed.
    import config
    import experiment1_discriminative_power
    import experiment2_consensus_centrality
    import experiment3_script_sensitivity
    import experiment4_null_output_validity
    import experiment5_cross_family_comparison

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    config.print_config_summary()
    config.validate_config()

    if args.dry_run:
        print("Dry run complete. Configuration verified.")
        return

    t0 = time.time()
    logger.info(f"Starting QRA Meta-Evaluation Execution Pipeline for WMT{config.YEAR}...")

    experiment1_discriminative_power.run()
    experiment2_consensus_centrality.run()
    experiment3_script_sensitivity.run()
    experiment4_null_output_validity.run()
    experiment5_cross_family_comparison.run()

    generate_narrative_reports(config)

    logger.info(f"Pipeline executed successfully in {time.time() - t0:.2f}s. "
                f"Outputs in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
