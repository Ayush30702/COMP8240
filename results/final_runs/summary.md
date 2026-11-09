# Final runs: summary

Environment: python 3.13.16, torch 2.14.1+cu130, transformers 5.22.1, spacy 3.8.18, numpy 2.4.6, scipy 1.16.3, scikit-learn 1.6.1, gpu Tesla T4, cpu_cores 2, ram_gb 13.6, multiprocessing_start_method fork

## Vocabularies (the authors' construction: train + dev, spaCy blank English tokenizer)

| Dataset | Word types | With GloVe vector | Without (replaced uniformly at random) | of which NFD mismatch | Sensitive, w = 0.9 | Sensitive with vector |
|---|---|---|---|---|---|---|
| sst2 | 14,700 | 14,283 | 417 | 28 | 13,231 | 12,816 |
| dreaddit | 14,840 | 14,256 | 584 | 4 | 13,357 | 12,778 |

## Tokens left unchanged by sanitization (dev split)

rare 90% = the words SanText+ treats as sensitive (w = 0.9); frequent 10% = the rest; no vector = words GloVe lacks, replaced uniformly at random.

| Dataset | Method | ε | Seed | Docs aligned | All | Content words | Rare 90% | Frequent 10% | No vector |
|---|---|---|---|---|---|---|---|---|---|
| dreaddit | SanText | 1 | 42 | 715/715 | 0.3% | 0.3% | 0.4% | 0.3% | 0.0% |
| dreaddit | SanText | 2 | 42 | 715/715 | 10.2% | 12.6% | 21.7% | 8.9% | 0.0% |
| dreaddit | SanText | 3 | 42 | 715/715 | 66.7% | 71.4% | 87.9% | 64.2% | 0.0% |
| dreaddit | SanText | 3 | 43 | 715/715 | 66.1% | 70.7% | 87.4% | 63.6% | 0.0% |
| dreaddit | SanText | 3 | 44 | 715/715 | 66.4% | 71.3% | 88.0% | 63.9% | 0.0% |
| dreaddit | SanText+ | 1 | 42 | 715/715 | 62.3% | 54.2% | 0.6% | 70.0% | 0.0% |
| dreaddit | SanText+ | 2 | 42 | 715/715 | 64.9% | 60.2% | 24.9% | 70.0% | 0.0% |
| dreaddit | SanText+ | 3 | 42 | 715/715 | 72.0% | 74.7% | 90.1% | 69.9% | 0.0% |
| dreaddit | SanText+ | 3 | 43 | 715/715 | 72.1% | 74.3% | 90.2% | 70.0% | 0.0% |
| dreaddit | SanText+ | 3 | 44 | 715/715 | 72.2% | 74.6% | 90.3% | 70.2% | 0.0% |
| sst2 | SanText | 1 | 42 | 872/872 | 0.3% | 0.4% | 0.4% | 0.3% | 0.0% |
| sst2 | SanText | 1 | 43 | 872/872 | 0.4% | 0.5% | 0.6% | 0.4% | 0.0% |
| sst2 | SanText | 1 | 44 | 872/872 | 0.4% | 0.4% | 0.5% | 0.4% | 0.0% |
| sst2 | SanText | 2 | 42 | 872/872 | 13.7% | 17.0% | 22.1% | 11.6% | 0.0% |
| sst2 | SanText | 2 | 43 | 872/872 | 13.2% | 16.7% | 21.9% | 11.0% | 0.0% |
| sst2 | SanText | 2 | 44 | 872/872 | 13.1% | 16.8% | 22.1% | 10.8% | 0.0% |
| sst2 | SanText | 3 | 42 | 872/872 | 74.3% | 79.8% | 89.8% | 70.6% | 0.0% |
| sst2 | SanText | 3 | 43 | 872/872 | 72.9% | 78.6% | 89.1% | 69.0% | 0.0% |
| sst2 | SanText | 3 | 44 | 872/872 | 73.2% | 78.7% | 88.6% | 69.5% | 0.0% |
| sst2 | SanText+ | 1 | 42 | 872/872 | 55.2% | 42.6% | 0.4% | 69.8% | 0.0% |
| sst2 | SanText+ | 1 | 43 | 872/872 | 55.2% | 43.3% | 0.6% | 69.8% | 0.0% |
| sst2 | SanText+ | 1 | 44 | 872/872 | 55.3% | 43.2% | 0.5% | 70.0% | 0.0% |
| sst2 | SanText+ | 2 | 42 | 872/872 | 60.2% | 52.2% | 24.9% | 69.8% | 0.0% |
| sst2 | SanText+ | 2 | 43 | 872/872 | 60.9% | 53.0% | 26.0% | 70.3% | 0.0% |
| sst2 | SanText+ | 2 | 44 | 872/872 | 60.9% | 52.7% | 24.8% | 70.7% | 0.0% |
| sst2 | SanText+ | 3 | 42 | 872/872 | 74.4% | 77.8% | 91.4% | 70.4% | 0.0% |
| sst2 | SanText+ | 3 | 43 | 872/872 | 74.2% | 77.6% | 91.1% | 70.2% | 0.0% |
| sst2 | SanText+ | 3 | 44 | 872/872 | 74.4% | 78.0% | 91.5% | 70.3% | 0.0% |

## Sanitization resources

| Job | Words | Rows with vector | Sensitive columns | Wall (s) | Peak memory (GB) |
|---|---|---|---|---|---|
| dreaddit__SanText__eps1.00__seed42 | 14840 | 14256 | 14256 | 113 | 5.30 |
| dreaddit__SanText__eps2.00__seed42 | 14840 | 14256 | 14256 | 109 | 5.31 |
| dreaddit__SanText__eps3.00__seed42 | 14840 | 14256 | 14256 | 105 | 5.31 |
| dreaddit__SanText_plus__eps1.00__seed42 | 14840 | 14256 | 12778 | 60 | 4.89 |
| dreaddit__SanText_plus__eps2.00__seed42 | 14840 | 14256 | 12778 | 59 | 4.90 |
| dreaddit__SanText_plus__eps3.00__seed42 | 14840 | 14256 | 12778 | 58 | 4.90 |
| sst2__SanText_plus__eps1.00__seed42 | 14700 | 14283 | 12816 | 97 | 4.92 |
| sst2__SanText_plus__eps2.00__seed42 | 14700 | 14283 | 12816 | 99 | 4.92 |
| sst2__SanText_plus__eps3.00__seed42 | 14700 | 14283 | 12816 | 105 | 4.91 |
| sst2__SanText__eps1.00__seed42 | 14700 | 14283 | 14283 | 199 | 5.32 |
| sst2__SanText__eps2.00__seed42 | 14700 | 14283 | 14283 | 194 | 5.32 |
| sst2__SanText__eps3.00__seed42 | 14700 | 14283 | 14283 | 185 | 5.31 |
| sst2__SanText__eps1.00__seed43 | 14700 | 14283 | 14283 | 192 | 5.32 |
| sst2__SanText__eps2.00__seed43 | 14700 | 14283 | 14283 | 204 | 5.33 |
| sst2__SanText__eps3.00__seed43 | 14700 | 14283 | 14283 | 196 | 5.33 |
| sst2__SanText__eps1.00__seed44 | 14700 | 14283 | 14283 | 192 | 5.33 |
| sst2__SanText__eps2.00__seed44 | 14700 | 14283 | 14283 | 195 | 5.32 |
| sst2__SanText__eps3.00__seed44 | 14700 | 14283 | 14283 | 197 | 5.31 |
| sst2__SanText_plus__eps1.00__seed43 | 14700 | 14283 | 12816 | 109 | 4.92 |
| sst2__SanText_plus__eps2.00__seed43 | 14700 | 14283 | 12816 | 107 | 4.91 |
| sst2__SanText_plus__eps3.00__seed43 | 14700 | 14283 | 12816 | 100 | 4.92 |
| sst2__SanText_plus__eps1.00__seed44 | 14700 | 14283 | 12816 | 104 | 4.92 |
| sst2__SanText_plus__eps2.00__seed44 | 14700 | 14283 | 12816 | 98 | 4.91 |
| sst2__SanText_plus__eps3.00__seed44 | 14700 | 14283 | 12816 | 99 | 4.92 |
| dreaddit__SanText__eps3.00__seed43 | 14840 | 14256 | 14256 | 104 | 5.31 |
| dreaddit__SanText__eps3.00__seed44 | 14840 | 14256 | 14256 | 105 | 5.31 |
| dreaddit__SanText_plus__eps3.00__seed43 | 14840 | 14256 | 12778 | 57 | 4.89 |
| dreaddit__SanText_plus__eps3.00__seed44 | 14840 | 14256 | 12778 | 59 | 4.90 |

## SST-2 dev accuracy against the paper (Yue et al., 2021, Table 1, GloVe)

| Method | ε | Paper | Ours, per seed | Mean | SD | Mean − paper |
|---|---|---|---|---|---|---|
| Unsanitized | – | 0.9251 | 0.9266 (s42), 0.9209 (s43) | 0.9237 | 0.0041 | -0.0014 |
| SanText | 1 | 0.5101 | 0.5057 (s42) | 0.5057 | – | -0.0044 |
| SanText | 2 | 0.5838 | 0.5940 (s42) | 0.5940 | – | +0.0102 |
| SanText | 3 | 0.8374 | 0.8326 (s42) | 0.8326 | – | -0.0048 |
| SanText+ | 1 | 0.7796 | 0.7833 (s42) | 0.7833 | – | +0.0037 |
| SanText+ | 2 | 0.7943 | 0.8016 (s42) | 0.8016 | – | +0.0073 |
| SanText+ | 3 | 0.8516 | 0.8486 (s42) | 0.8486 | – | -0.0030 |

## New datasets (dev = held-out test split)

| Dataset | Method | ε | Seed | Accuracy | Macro-F1 | F1 (label 1) |
|---|---|---|---|---|---|---|
| dreaddit | Unsanitized | – | 42 | 0.8056 | 0.8042 | 0.8206 |
| dreaddit | SanText | 1 | 42 | 0.5343 | 0.5336 | 0.5518 |
| dreaddit | SanText | 2 | 42 | 0.5469 | 0.5468 | 0.5500 |
| dreaddit | SanText | 3 | 42 | 0.7441 | 0.7416 | 0.7669 |
| dreaddit | SanText+ | 1 | 42 | 0.7441 | 0.7414 | 0.7675 |
| dreaddit | SanText+ | 2 | 42 | 0.7287 | 0.7260 | 0.7532 |
| dreaddit | SanText+ | 3 | 42 | 0.7441 | 0.7419 | 0.7657 |

