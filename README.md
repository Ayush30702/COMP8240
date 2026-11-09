# COMP8240
# Replicating SanText and testing it on sensitive health and banking text

COMP8240 Applications of Data Science, Macquarie University (2026) · Ayush Mahesh Ghore

This repository accompanies the final report (`report/`). It replicates **SanText** (Yue et al., 2021, *Differential Privacy for Text Analytics via Natural Text Sanitization*, Findings of ACL-IJCNLP 2021; [paper](https://aclanthology.org/2021.findings-acl.337), [code](https://github.com/xiangyue9607/SanText)) and applies it to two datasets with real privacy stakes: Dreaddit and a new dataset of Australian health- and banking-app reviews.

## Layout

| Folder | Contents |
|---|---|
| `report/` | LaTeX source of the report (ACL template) and the final PDF |
| `notebooks/` | Four Colab notebooks that reproduce the results (see below) |
| `scripts/` | Everything the notebooks run, as standalone, documented scripts |
| `data/` | The review test split, its annotation and guidelines, and a data statement (`data/README.md`) |
| `results/` | Metrics of every run, summary tables, and the attack and LLM-judge outputs (scores only, no text) |

Main scripts:
- `build_reviews_dataset.py`: collects, cleans, labels and splits the review dataset, and draws the annotation sample.
- `finetune_sanitized.py`: BERT fine-tuning with the paper's settings. The authors' `run_glue.py` needs Transformers 3.3.1, which no longer installs.
- `santext_chunked.py`: SanText with the authors' own probability function, computed a block of rows at a time. It reaches the paper's BERT-embedding setting (about 19 GB with the released code) and QNLI (about 160 GB) in under 4 GB of memory.
- `mask_attack.py`: the paper's mask token inference attack, both as released (inputs swapped) and fixed.
- `llm_judge.py` and `build_judge_items.py`: the local LLM judge (Qwen2.5-7B-Instruct, 4-bit) and its items. `prepare_reviews_judge.py` builds the reviews' items from the 9 October outputs; `analyse_reviews_judge.py` scores them and the human check.
- `filter_glove.py`, `sanitize_queue.py`, `make_plan.py`, `analyse_runs.py`, `analyse_c.py`, `survival_gold.py`: run orchestration and analysis.

## Reproducing the results (Google Colab, T4 GPU)

The authors' code is cloned at commit `9c35f6e`; it is not copied here. SST-2, QNLI and Dreaddit are downloaded by the notebooks.

1. `notebooks/COMP8240_SanText_update.ipynb` is the 9 October run. It built the review dataset, sampled the annotation set, ran SanText on SST-2 (run 1 in the report) and made the reviews' only sanitized run (SanText, ε = 3).
2. `notebooks/COMP8240_final_runs.ipynb` runs SanText and SanText+ at ε = 1, 2, 3 on SST-2 and Dreaddit, plus reproducibility checks. Its review jobs need the raw training split, which was lost (see `data/README.md`).
3. `notebooks/COMP8240_privacy_and_judge.ipynb` covers the dataset statistics, SanText with BERT embeddings, the attack and the LLM judge on SST-2 and Dreaddit.
4. `notebooks/COMP8240_reviews_judge.ipynb` runs the same judge on the reviews. Its items are embedded, so it needs no uploads.

Notebooks 2 and 3 save to `MyDrive/COMP8240_final`, skip finished steps after a disconnect, and finish by downloading a zip of their outputs.

## Main results

| | Paper | Ours |
|---|---|---|
| SST-2, SanText+ ε = 1 / 2 / 3 (accuracy) | 0.780 / 0.794 / 0.852 | 0.783 / 0.802 / 0.849 |
| SST-2, SanText ε = 3, two runs with seed 42 | 0.837 | 0.851, 0.833 |
| Defense rate gain at ε = 16 (BERT embeddings) | "20%" | +21 points |
| Accuracy loss at ε = 16 | "4%" (5.5 points in its Table 3) | 8.3 points |
| Dreaddit stress F1, unsanitized → SanText ε = 3 | (new) | 0.821 → 0.767 |
| Reviews accuracy, unsanitized → SanText ε = 3 | (new) | 0.906 → 0.854 |
| Flagged disclosure words left unchanged by SanText at ε = 3 | (new) | 41 of 48 (85%) |

See the report for the full results, the eight problems we found in the released code, and the ethics discussion.

## Licence and use

Our code is released under the MIT licence (`LICENSE`). The review data is released for research use only: see `data/README.md` before using it.

## Use of AI assistants

Claude (Anthropic) helped write and debug the code and draft the report, and served as the second annotator. The author labelled annotator A's data by hand, ran all experiments and checked every number and reference.
