"""
config.py — Central Configuration & Experimental Parameters
─────────────────────────────────────────────────────────────────────────────
Defines directory layouts, dataset selection (WMT25 / WMT26), language pairs,
coverage tiers, system-type mapping, instrument directionality, and global
statistical constants.
"""

import os

# ─── Dataset Selection ──────────────────────────────────────────────────────
YEAR = int(os.environ.get("WMT_YEAR", 2025))
if YEAR not in (2025, 2026):
    raise ValueError(f"config.YEAR must be 2025 or 2026, got {YEAR}")

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Outputs are namespaced per year so switching YEAR never overwrites the
# other year's results — both can be produced and compared side by side.
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", f"wmt{YEAR}")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

for _d in (DATA_DIR, OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── Datasets ───────────────────────────────────────────────────────────────
_YY = YEAR % 100  # file naming convention used by the data-prep scripts: wmt26_*, wmt25_*
RESULTS_CSV = os.path.join(DATA_DIR, f"wmt{_YY}_results.csv")
SEGMENT_RESULTS_CSV = os.path.join(DATA_DIR, f"wmt{_YY}_segment_results.csv")
JUDGE_SCORES_CSV = os.path.join(DATA_DIR, f"wmt{_YY}_judge_scores.csv")
JUDGE_SEGMENT_SCORES_CSV = os.path.join(DATA_DIR, f"wmt{_YY}_judge_segment_scores.csv")
HUMAN_SCORES_CSV = os.path.join(DATA_DIR, f"wmt{_YY}_human_scores.csv")
HUMAN_SEGMENT_SCORES_CSV = os.path.join(DATA_DIR, f"wmt{_YY}_human_segment_scores.csv")

ALLOW_SYNTHETIC_FALLBACK = bool(int(os.environ.get("WMT_ALLOW_SYNTHETIC_FALLBACK", "0")))

# ─── Instruments & Directionality (year-dependent) ─────────────────────────
if YEAR == 2026:
    METRIC_INSTRUMENTS = ["BLEU", "METEOR", "TER", "CHRF++", "BERTScore", "COMET"]
else:  # 2025
    METRIC_INSTRUMENTS = ["BLEU", "METEOR", "ROUGE-L", "chrF", "TER", "Cos Similarity"]

HIGHER_IS_BETTER = {
    "BLEU": True, "METEOR": True, "TER": False, "ROUGE-L": True,
    "CHRF++": True, "chrF": True, "BERTScore": True, "COMET": True,
    "Cos Similarity": True,
}

LLM_JUDGE_INSTRUMENTS = []   # done in seperate branch 
HUMAN_INSTRUMENTS = []       # 

if YEAR == 2026:
    NARRATIVE_B_NEURAL_METRIC = "COMET"
    NARRATIVE_B_SURFACE_METRIC = "CHRF++"
else:
    NARRATIVE_B_NEURAL_METRIC = "Cos Similarity"
    NARRATIVE_B_SURFACE_METRIC = "chrF"

# ─── Thresholds & Protocols ────────────────────────────────────────────────
DEGENERATE_BLEU_MAX = 1.0
PRIMARY_SYSTEM_TYPES = {"primary"}
"""Unit of analysis for every ranking-level statistic (CV*, correlations,
Kendall's W, pairwise P, floor-inflation). Enforced in
data_loader.wide_by_instrument()."""

BOOTSTRAP_B = 1000
PERMUTATION_N = 10000
ALPHA = 0.05
RANDOM_SEED = 16

# ─── Language Pairs ─────────────────────────────────────────────────────────
if YEAR == 2026:
    LANGUAGE_PAIRS = [
        "English to Assamese", "Assamese to English",
        "English to Mizo", "Mizo to English",
        "English to Khasi", "Khasi to English",
        "English to Manipuri (Bengali)", "Manipuri(Bengali) to English",
        "English to Manipuri(Meitei Mayek)", "Manipuri(Meitei Mayek) to English",
        "English to Bodo", "Bodo to English",
        "English to Kokborok", "Kokborok to English",
        "English to Karbi", "Karbi to English",
        "English to Nagamese", "Nagamese to English",
        "English to Tagin", "Tagin to English",
    ]
else:  
    LANGUAGE_PAIRS = [
        "English to Assamese", "Assamese to English",
        "English to Manipuri", "Manipuri to English",
        "English to Khasi", "Khasi to English",
        "English to Mizo", "Mizo to English",
        "English to Nyishi", "Nyishi to English",
        "English to Bodo", "Bodo to English",
        "English to Kokborok", "Kokborok to English",
    ]

# ─── Experiment 3 Script Sensitivity Definitions ────────────────────────────
# Only WMT26 differentiates Manipuri by script (Bengali vs Meitei Mayek).
if YEAR == 2026:
    SCRIPT_PAIR_DIRECTIONS = [
        {
            "direction": "en2indic",
            "meitei_pair": "English to Manipuri(Meitei Mayek)",
            "bengali_pair": "English to Manipuri (Bengali)",
        },
        {
            "direction": "indic2en",
            "meitei_pair": "Manipuri(Meitei Mayek) to English",
            "bengali_pair": "Manipuri(Bengali) to English",
        },
    ]
else:
    SCRIPT_PAIR_DIRECTIONS = []

# ─── Coverage Tiers (Experiment 3 extension) ────────────────────────────────
if YEAR == 2026:
    TIER_ASSIGNMENT = {
        "Assamese": {"tier": 1, "script": "Bengali", "status": "VERIFIED_XLM_R"},
        "Bodo": {"tier": 2, "script": "Devanagari", "status": "VERIFIED_XLM_R"},
        "Manipuri (Bengali)": {"tier": 2, "script": "Bengali", "status": "VERIFIED_XLM_R"},
        "Manipuri (Meitei Mayek)": {"tier": 3, "script": "Meitei Mayek", "status": "VERIFIED_XLM_R_UNCOVERED"},
        "Mizo": {"tier": 4, "script": "Latin", "status": "VERIFIED_ABSENT"},
        "Khasi": {"tier": 4, "script": "Latin", "status": "VERIFIED_ABSENT"},
        "Kokborok": {"tier": 4, "script": "Latin", "status": "VERIFIED_ABSENT"},
        "Karbi": {"tier": 4, "script": "Latin", "status": "VERIFIED_ABSENT"},
        "Tagin": {"tier": 4, "script": "Latin", "status": "VERIFIED_ABSENT"},
        "Nagamese": {"tier": 4, "script": "Latin", "status": "VERIFIED_ABSENT"},
    }
    PAIR_TO_LANGUAGE = {
        "English to Assamese": "Assamese", "Assamese to English": "Assamese",
        "English to Mizo": "Mizo", "Mizo to English": "Mizo",
        "English to Khasi": "Khasi", "Khasi to English": "Khasi",
        "English to Manipuri (Bengali)": "Manipuri (Bengali)", "Manipuri(Bengali) to English": "Manipuri (Bengali)",
        "English to Manipuri(Meitei Mayek)": "Manipuri (Meitei Mayek)", "Manipuri(Meitei Mayek) to English": "Manipuri (Meitei Mayek)",
        "English to Bodo": "Bodo", "Bodo to English": "Bodo",
        "English to Kokborok": "Kokborok", "Kokborok to English": "Kokborok",
        "English to Karbi": "Karbi", "Karbi to English": "Karbi",
        "English to Nagamese": "Nagamese", "Nagamese to English": "Nagamese",
        "English to Tagin": "Tagin", "Tagin to English": "Tagin",
    }
else:  # 2025
    TIER_ASSIGNMENT = {
        "Assamese": {"tier": 1, "script": "Bengali", "status": "NEEDS_VERIFICATION"},
        "Bodo": {"tier": 2, "script": "Devanagari", "status": "NEEDS_VERIFICATION"},
        "Manipuri": {"tier": 2, "script": "Bengali (assumed)", "status": "NEEDS_VERIFICATION"},
        "Khasi": {"tier": 4, "script": "Latin", "status": "NEEDS_VERIFICATION"},
        "Mizo": {"tier": 4, "script": "Latin", "status": "NEEDS_VERIFICATION"},
        "Nyishi": {"tier": 4, "script": "Latin", "status": "NEEDS_VERIFICATION"},
        "Kokborok": {"tier": 4, "script": "Latin", "status": "NEEDS_VERIFICATION"},
    }
    PAIR_TO_LANGUAGE = {
        "English to Assamese": "Assamese", "Assamese to English": "Assamese",
        "English to Manipuri": "Manipuri", "Manipuri to English": "Manipuri",
        "English to Khasi": "Khasi", "Khasi to English": "Khasi",
        "English to Mizo": "Mizo", "Mizo to English": "Mizo",
        "English to Nyishi": "Nyishi", "Nyishi to English": "Nyishi",
        "English to Bodo": "Bodo", "Bodo to English": "Bodo",
        "English to Kokborok": "Kokborok", "Kokborok to English": "Kokborok",
    }


def tier_for_pair(pair: str):
    lang = PAIR_TO_LANGUAGE.get(pair)
    return TIER_ASSIGNMENT.get(lang, {}).get("tier") if lang else None


# ─── System-Type Taxonomy (Narrative b: Circularity Hypothesis) ─────────────
SYSTEM_BACKBONE_CATEGORY = {
    "indictrans2": "Multilingual Pretrained (IndicTrans2)",
    "nllb": "Multilingual Pretrained (NLLB)",
    "sarvam": "Multilingual Pretrained (Sarvam)",
    "scratch_transformer": "Scratch/Custom Transformer",
    "rule_based": "Rule-Based/Non-Neural",
}


def print_config_summary():
    print(
        f"[config] YEAR={YEAR} | RESULTS_CSV={RESULTS_CSV}\n"
        f"[config] {len(LANGUAGE_PAIRS)} language pairs | {len(METRIC_INSTRUMENTS)} metric instruments "
        f"({', '.join(METRIC_INSTRUMENTS)})\n"
        f"[config] script-sensitivity (Exp3, dual-script Manipuri): "
        f"{'enabled' if SCRIPT_PAIR_DIRECTIONS else 'disabled — not available for this year'}\n"
        f"[config] LLM judges: {len(LLM_JUDGE_INSTRUMENTS)} | human annotations: {len(HUMAN_INSTRUMENTS)}"
    )


def validate_config() -> bool:
    errors = []
    if not METRIC_INSTRUMENTS:
        errors.append("ERROR: METRIC_INSTRUMENTS is empty")
    for m in METRIC_INSTRUMENTS:
        if m not in HIGHER_IS_BETTER:
            errors.append(f"WARNING: {m} not in HIGHER_IS_BETTER mapping")
    if not LANGUAGE_PAIRS:
        errors.append("ERROR: LANGUAGE_PAIRS is empty")
    for pair in LANGUAGE_PAIRS:
        lang = PAIR_TO_LANGUAGE.get(pair)
        if lang and lang not in TIER_ASSIGNMENT:
            errors.append(f"WARNING: {pair} -> {lang} not in TIER_ASSIGNMENT")
        if not lang:
            errors.append(f"WARNING: {pair} missing from PAIR_TO_LANGUAGE")
    if errors:
        for e in errors:
            print(e)
    return len([e for e in errors if e.startswith("ERROR")]) == 0


if __name__ == "__main__":
    print_config_summary()
    print("\nConfiguration valid." if validate_config() else "\nConfiguration has errors — see above.")
