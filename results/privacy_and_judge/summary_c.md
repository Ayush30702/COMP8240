# Notebook C: summary

Environment: python 3.13.16, torch 2.14.1+cu130, transformers 5.22.1, spacy 3.8.18, numpy 2.4.6, scipy 1.16.3, scikit-learn 1.6.1, gpu Tesla T4, cpu_cores 2, ram_gb 13.6

## Original datasets, measured from the files

**SST-2**: train_rows: 67349; header: sentence	label; train_labels: {'0': 29780, '1': 37569}; train_words_mean: 9.4; dev_rows: 872; dev_labels: {'1': 444, '0': 428}; dev_words_mean: 19.5; test_rows: 1821; vocabulary_authors_construction: 14700
  - `hide new secretions from the parental units 	0`
  - `contains no wit , only labored gags 	0`
  - `it 's a charming and often affecting journey . 	1`

**QNLI**: train_rows: 104743; header: index	question	sentence	label; train_labels: {'not_entailment': 52366, 'entailment': 52377}; train_question_words_mean: 9.9; train_sentence_words_mean: 26.5; dev_rows: 5463; dev_labels: {'entailment': 2702, 'not_entailment': 2761}; dev_question_words_mean: 10.0; dev_sentence_words_mean: 27.6; test_rows: 5463; vocabulary_authors_construction: 87751
  - `0	When did the third Digimon series begin?	Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese.	not_entailment`
  - `1	Which missile batteries often have individual launchers several kilometres from one another?	When MANPADS is operated by specialists, batteries may have several dozen teams deploying separately in small sections; self-propelled air defence guns may deploy in pairs.	not_entailment`
  - `0	What came into force after the new constitution was herald?	As of that day, the new constitution heralding the Second Republic came into force.	entailment`

## Block-wise sanitizer vs the authors' code (SST-2, GloVe, SanText, seed 42)

| ε | Rows checked | Max abs. difference | Rows bit-identical | Unchanged tokens (dev), block-wise | Unchanged tokens (dev), authors' runs |
|---|---|---|---|---|---|
| 1 | 2000 | 1.9e-09 | 1253 | 0.0038 | 0.0033 (notebook A/B) |
| 2 | 2000 | 8.9e-08 | 1244 | 0.1356 | 0.1368 (notebook A/B) |
| 3 | 2000 | 3.6e-07 | 1304 | 0.7363 | 0.7426 (notebook A/B) |

## SanText with BERT embeddings on SST-2 (dev): utility and the mask token inference attack

| ε | Tokens unchanged | Accuracy (ours) | Accuracy (paper, Table 3) | Defense rate, attack as released | Defense rate, attack fixed | Positions (skipped as released) | Sanitizer s / peak GB |
|---|---|---|---|---|---|---|---|
| ∞ (unsanitized) | – | 0.9266 | 0.9251 | 0.395 | 0.395 | 20199 (0) | – / – |
| 8 | 0.013 | – | – | 0.955 | 0.953 | 20199 (0) | 113.9 / 1.76 |
| 12 | 0.201 | – | 0.6084 | 0.920 | 0.920 | 20199 (0) | 92.2 / 1.75 |
| 14 | 0.497 | – | 0.7548 | 0.811 | 0.810 | 20199 (0) | 47.3 / 1.75 |
| 16 | 0.782 | 0.8440 | 0.8698 | 0.606 | 0.604 | 20199 (0) | 44.0 / 1.75 |
| 20 | 0.979 | – | – | 0.417 | 0.414 | 20199 (0) | 43.7 / 1.75 |

## LLM judge (Qwen/Qwen2.5-7B-Instruct (4-bit NF4), prompt v1)

Probability on the allowed answer tokens (format compliance): meaning 1.000

### Meaning preserved (1–5, expected value)

| Dataset | Method | ε | n | Mean | SD |
|---|---|---|---|---|---|
| dreaddit | SanText | 1 | 100 | 1.00 | 0.00 |
| dreaddit | SanText | 2 | 100 | 1.01 | 0.06 |
| dreaddit | SanText | 3 | 100 | 2.21 | 0.42 |
| dreaddit | SanText+ | 1 | 100 | 1.60 | 0.42 |
| dreaddit | SanText+ | 2 | 100 | 1.83 | 0.38 |
| dreaddit | SanText+ | 3 | 100 | 2.11 | 0.40 |
| dreaddit | identical | – | 20 | 5.00 | 0.00 |
| sst2 | SanText | 1 | 100 | 1.02 | 0.09 |
| sst2 | SanText | 2 | 100 | 1.07 | 0.20 |
| sst2 | SanText | 3 | 100 | 2.70 | 0.91 |
| sst2 | SanText+ | 1 | 100 | 1.69 | 0.63 |
| sst2 | SanText+ | 2 | 100 | 1.93 | 0.70 |
| sst2 | SanText+ | 3 | 100 | 2.68 | 0.99 |
| sst2 | identical | – | 20 | 5.00 | 0.00 |

### Is a disclosure still judged present after sanitization? (reviews)

| Version | ε | Gold-disclosing reviews still judged disclosing | Non-disclosing reviews judged disclosing |
|---|---|---|---|

### Targeted check: does the sanitized review still reveal the flagged information?

| Version | ε | n | Still reveals | Mean P(yes) |
|---|---|---|---|---|

