"""
data_loader.py — Unified Long/Wide Data Loader
─────────────────────────────────────────────────────────────────────────────
Handles system-level (and, once available, segment-level) score ingestion,
sign orientation, and wide pivots restricted to primary submissions.

"""

import os
import logging
import pandas as pd
import numpy as np
import config

logger = logging.getLogger(__name__)


def _generate_synthetic_data_if_missing():

    if os.path.exists(config.RESULTS_CSV):
        return

    if not config.ALLOW_SYNTHETIC_FALLBACK:
        raise FileNotFoundError(
            f"config.RESULTS_CSV not found: {config.RESULTS_CSV}\n"
            f"Place {os.path.basename(config.RESULTS_CSV)} in {config.DATA_DIR}/, or set "
            f"WMT_ALLOW_SYNTHETIC_FALLBACK=1 to smoke-test the pipeline with "
            f"fabricated data (never do this for a real analysis run)."
        )

    logger.warning(
        "ALLOW_SYNTHETIC_FALLBACK=1 and no real results file was found — "
        "generating FABRICATED data for a smoke test only. Do not report "
        "any numbers produced this way."
    )
    np.random.seed(config.RANDOM_SEED)

    teams = [f"Team_{c}" for c in "ABCDEFGH"]
    backbones = ["indictrans2", "nllb", "sarvam", "scratch_transformer",
                 "scratch_transformer", "rule_based", "indictrans2", "nllb"]

    sys_rows, seg_rows = [], []

    for pair in config.LANGUAGE_PAIRS:
        for idx, (team, bb) in enumerate(zip(teams, backbones)):
            sys_id = f"{team}_primary"
            is_degenerate = (idx == len(teams) - 1 and "English to" in pair)

            base_bleu = 0.5 if is_degenerate else np.random.uniform(12.0, 28.0)
            base_neural = 0.10 if is_degenerate else (
                base_bleu / 40.0 + (0.15 if "Multilingual" in config.SYSTEM_BACKBONE_CATEGORY.get(bb, "") else 0.05)
            )
            if "Meitei Mayek" in pair:
                base_neural += 0.12  # simulated neural-metric inflation on an unseen script

            scores = {i: np.nan for i in config.METRIC_INSTRUMENTS}
            for instr in config.METRIC_INSTRUMENTS:
                higher = config.HIGHER_IS_BETTER.get(instr, True)
                if instr in ("BLEU",):
                    scores[instr] = base_bleu
                elif instr in ("METEOR", "ROUGE-L"):
                    scores[instr] = min(base_bleu * 1.5, 100.0)
                elif instr == "TER":
                    scores[instr] = 98.0 if is_degenerate else max(0.0, 85.0 - base_bleu)
                elif instr in ("CHRF++", "chrF"):
                    scores[instr] = base_bleu * 2.1
                elif instr in ("BERTScore", "Cos Similarity"):
                    scores[instr] = 0.50 + base_bleu / 100.0
                elif instr == "COMET":
                    scores[instr] = base_neural
                else:
                    scores[instr] = base_bleu if higher else 100.0 - base_bleu

            for instr, val in scores.items():
                sys_rows.append({
                    "category": "",
                    "language_pair": pair,
                    "src_lang": pair.split(" to ")[0],
                    "tgt_lang": pair.split(" to ")[1],
                    "direction": "en2indic" if pair.startswith("English") else "indic2en",
                    "late_submission": False,
                    "system_type": "primary",
                    "team": team,
                    "system": sys_id,
                    "backbone": bb,
                    "instrument": instr,
                    "score": float(val),
                })

    df_sys = pd.DataFrame(sys_rows)
    df_sys.to_csv(config.RESULTS_CSV, index=False)
    df_sys[df_sys["instrument"].isin(config.LLM_JUDGE_INSTRUMENTS)].to_csv(config.JUDGE_SCORES_CSV, index=False)
    df_sys[df_sys["instrument"].isin(config.HUMAN_INSTRUMENTS)].to_csv(config.HUMAN_SCORES_CSV, index=False)
    pd.DataFrame(seg_rows).to_csv(config.SEGMENT_RESULTS_CSV, index=False)

    logger.info("Synthetic smoke-test dataset created.")


def load_all_scores(level: str = "system") -> pd.DataFrame:

    _generate_synthetic_data_if_missing()

    if level == "system":
        if not os.path.exists(config.RESULTS_CSV):
            raise FileNotFoundError(f"Required file not found: {config.RESULTS_CSV}")
        df = pd.read_csv(config.RESULTS_CSV)
    else:
        if not os.path.exists(config.SEGMENT_RESULTS_CSV):
            logger.warning(
                f"No segment-level data at {config.SEGMENT_RESULTS_CSV} for WMT{config.YEAR} "
                f"— segment-level analyses will be skipped for this run."
            )
            return pd.DataFrame(columns=["language_pair", "team", "system",
                                          "segment_id", "instrument", "score"])
        df = pd.read_csv(config.SEGMENT_RESULTS_CSV)

    if "language_pair" in df.columns:
        df["tier"] = df["language_pair"].map(config.tier_for_pair)
    return df


def oriented(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    def _orient(row):
        higher = config.HIGHER_IS_BETTER.get(row["instrument"], True)
        return row["score"] if higher else -row["score"]

    out["score"] = out.apply(_orient, axis=1)
    return out


def wide_by_instrument(df: pd.DataFrame, language_pair: str,
                        system_types=None) -> pd.DataFrame:

    system_types = system_types if system_types is not None else config.PRIMARY_SYSTEM_TYPES
    sub = df[df["language_pair"] == language_pair]
    if "system_type" in sub.columns:
        sub = sub[sub["system_type"].isin(system_types)]
    if sub.empty:
        return pd.DataFrame()

    idx_cols = ["language_pair", "team", "system"]
    for extra in ("backbone", "category"):

        if extra in sub.columns and sub[extra].notna().any():
            idx_cols.append(extra)

    wide = sub.pivot_table(index=idx_cols, columns="instrument", values="score", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide
