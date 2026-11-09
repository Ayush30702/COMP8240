# Data statement: Australian health and banking app reviews

**What it is.** 2,712 English Google Play reviews of eight Australian-market apps: four health apps (mySugr, Medisafe, MyTherapy, Headspace) and four banks (CommBank, Westpac, NAB, ANZ). Each review has a sentiment label taken from its star rating. A sample of 100 test reviews is also annotated for personal disclosures. The dataset tests text sanitization on text with real health and financial content.

**What this folder holds.**

| File | Contents |
|---|---|
| `app_reviews/dev.tsv` | The 542-review test split (SST-2 format: text, label). Two person names and one date of birth are replaced by `[NAME]` and `[DATE]`. |
| `app_reviews/stats.json` | The construction funnel and dataset statistics |
| `app_reviews/collection_log.json` | What was collected, when, and from which apps |
| `annotation/annotation_labels.csv` | Annotator A's disclosure labels, the gold labels, and the flagged words and their categories, keyed by a hash of each review. One flagged name is redacted as `[NAME]`. |
| `annotation/guidelines.md` | The annotation guidelines |

The raw training split (2,170 reviews) is not available: its only copy was in a cloud folder we lost access to. Its SanText version at ε = 3 survives, but we don't release it, because SanText at that ε leaves most disclosing words intact.

**Collection.** On 9 October 2026 we collected the 500 newest reviews per app (country `au`, language `en`) with `google-play-scraper`, using `scripts/build_reviews_dataset.py`. Reviewer names, avatars and developer replies were never stored.

**Processing.** Each step removed reviews in turn:

| Step | Total | Health | Banking |
|---|---|---|---|
| Collected | 4,000 | 2,000 | 2,000 |
| Exact duplicates removed | 3,712 | 1,951 | 1,761 |
| Shorter than five words removed | 3,127 | 1,726 | 1,401 |
| Non-English removed (word-list test) | 2,978 | 1,662 | 1,316 |
| Three-star and unrated removed | 2,712 | 1,493 | 1,219 |

Labels: 1–2 stars = 0 (negative), 4–5 stars = 1 (positive). The split is stratified by app type and label: 2,170 train and 542 test (`dev.tsv`). 41.3% of reviews are positive (banking 31%, health 50%). The median length is 23 words, the mean 32 and the maximum 183.

**Annotation.** Two annotators labelled 100 test reviews, stratified by app type and star label, following `annotation/guidelines.md`. Annotator A was the author, by hand; annotator B was Claude, an LLM, using the same guidelines independently. Agreement was κ = 0.88 for sentiment and κ = 0.56 for disclosure. The 10 disagreements were resolved by the guidelines. The gold labels mark 13 reviews as disclosing, with 48 flagged content words.

**Ethics and terms of use.** The reviews are public, but some disclose health or financial details, and one test review signs off with a full name and a date of birth. A review's text can also be matched to its author's public Google Play name. Use the data for research only, do not try to identify reviewers, and report results in aggregate. To request removal of a review, open an issue in this repository. A formal publication would also need to confirm that Google Play's terms allow redistribution. Third-party data (SST-2, QNLI, Dreaddit) is not redistributed here.
