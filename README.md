# which-metric-which-script

# QRA++ Meta-Evaluation Pipeline

A statistical meta-evaluation pipeline for **WMT Indic-MT** shared task results. It answers a simple question — *can we trust these automatic metrics to rank translation systems?* — by running five experiments against WMT25 and WMT26 data: discriminative power, cross-metric consensus, script sensitivity, null-output robustness, and cross-family validity (metrics vs. LLM judges vs. humans).

Supports both **WMT25** and **WMT26** out of the box — language pairs, metric sets, and coverage tiers switch automatically based on the year you run.

---

## Table of Contents

- [What it does](#what-it-does)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Output](#output)
- [Configuration](#configuration)
- [License](#license)

---

## What it does

| # | Experiment | Question it answers |
|---|------------|----------------------|
| 1 | **Discriminative Power** | Which metrics actually spread systems apart, instead of clustering them together? (CV*, range, effective distinct values) |
| 2 | **Consensus Centrality** | How much do metrics agree with each other, and which one sits closest to consensus? (Kendall's W, pairwise agreement, leave-one-out centrality) |
| 3 | **Script Sensitivity & Coverage Tiers** | Does script (e.g. Meitei Mayek vs. Bengali) or a language's model-coverage tier bias a metric's scores? |
| 4 | **Null-Output Validity** | Do metrics over-reward degenerate or empty-output systems? |
| 5 | **Cross-Family Comparison** | How well do automatic metrics, LLM judges, and human ratings agree — and which one is closer to human judgment? (Williams test, permutation tests) |

`main.py` runs all five in order, then synthesizes two narrative text reports directly from the computed tables.

---

## Project structure

```
.
├── main.py                                  # entry point — orchestrates the full pipeline
├── config.py                                # paths, language pairs, thresholds, year switch
├── data_loader.py                           # CSV ingestion, sign orientation, wide pivots
├── stats_utils.py                           # CV*, Kendall's W, pairwise P, bootstrap, permutation tests
├── plot_utils.py                            # shared heatmap / bar chart / radar chart helpers
├── experiment1_discriminative_power.py
├── experiment2_consensus_centrality.py
├── experiment3_script_sensitivity.py
├── experiment4_null_output_validity.py
├── experiment5_cross_family_comparison.py
├── requirements.txt
├── data/                                    # you add this — see below
└── outputs/
    └── wmt{year}/
        ├── tables/                          # every result as CSV
        ├── figures/                         # every plot as PNG
        ├── narrative_a_summary.txt
        └── narrative_b_summary.txt
```

---

## Installation

```bash
git clone <repo-url>
cd <repo-name>
```

Requires Python 3.9+.

---

## Data

Create a `data/` folder at the project root and add the results file for the year you're running:

```
data/wmt25_results.csv   # for WMT25
data/wmt26_results.csv   # for WMT26
```
---

## Usage

Run the full pipeline for a given year:

```bash
python main.py --year 2025
```

```bash
python main.py --year 2026
```

Validate your setup without running any analysis:

```bash
python main.py --year 2025 --dry-run
```

---

## Output

Everything is namespaced by year, so both can be run side by side without overwriting each other:

```
outputs/wmt{year}/
├── tables/                      # CSV results for all 5 experiments
├── figures/                     # heatmaps, bar charts, and radar charts
├── narrative_a_summary.txt      # metric trustworthiness & script coverage synthesis
└── narrative_b_summary.txt      # system standing & circularity hypothesis synthesis
```

---

## Configuration

All tunable behavior lives in `config.py`:

| Setting | Purpose |
|---|---|
| `WMT_YEAR` (env var) or `--year` | Switches language pairs, metric sets, and tier maps between WMT25/WMT26 |
| `HIGHER_IS_BETTER` | Score direction per instrument (e.g. TER is lower-is-better) |
| `BOOTSTRAP_B`, `PERMUTATION_N` | Bootstrap resamples / permutation iterations for statistical tests |
| `DEGENERATE_BLEU_MAX` | BLEU threshold for flagging a degenerate/null-output system |

Run `python config.py` to print and validate the active configuration without executing the pipeline.

---
```
@inproceedings{huidrom-etal-2026-metric,
  title     = {Which Metric for Which Script? A Quantified Meta-Evaluation of Automatic Metrics and LLM Judges for Low-Resource Indic Machine Translation},
  author    = {Huidrom, Rudali and Kumar, Vikas and Pangsatabam, Hoomexsun and Das, Pinaki and Singh, Kshetrimayum Boynao and Konjengbam, Anand},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation (WMT26)},
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}
```
