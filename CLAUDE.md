# PromptOutputClassifier

## Scope
Predict how many tokens a chat model will generate for a user prompt, so an inference scheduler can run
short requests first and long ones later. The current stage is a **quick MVP**: get a simple, fast model
running end to end and measure it. It is not yet a production scheduler, and there is no inference API or saved model.

Current target: **claude-1, English, first-turn prompts only**, predicting response length as quantiles (p25/p50/p75/p90).

## Environment
- Windows 11, Python 3.13, no virtualenv or requirements file. Packages used: `pandas`, `pyarrow`, `tiktoken`, `scikit-learn`.
- Run: `python main.py` (about 1.5-2 minutes; prints score tables and example predictions).
- Not a git repository.

## Data (`data/`)
LMSYS-Chat-1M from Hugging Face: 6 parquet shards (~1.5 GB), 1M conversations, 25 models.
- Raw columns: `conversation_id`, `model`, `conversation` (list of `{role, content}` messages), `turn`, `language`,
  `openai_moderation` (large, not loaded), `redacted`.
- vicuna-13b is about half the data; ~2/3 of conversations are single-turn; ~27% are redacted (placeholders like `NAME_1`).
- English first-turn pairs per GPT/Claude model: claude-1 13,805 · gpt-3.5-turbo 6,483 · gpt-4 6,303 ·
  claude-instant-1 4,589 · claude-2 1,881.
- Responses appear capped by the arena: most models' 99th percentile is 450-900 tokens; claude-1 rarely exceeds ~600.
  A few open models have runaway outputs (up to 400k tokens) and ~1% of responses are empty.

## Code
- `lmsys_adapter.py` - `LmsysAdapter(data_dir, max_rows=None, seed=0, **filters)` loads everything in one call:
  read with filters, explode conversations into prompt/response pairs, count tokens, drop empty responses,
  shuffle, cap, and add a `bucket` column. Helper methods (`filter_eq`, `filter_range`, `sort`, `sample`) return
  plain DataFrames. Also defines the fixed `BUCKETS` / `BUCKET_EDGES`.
- `main.py` - loads one model's data, splits 80/20 by position, trains quantile regression models, fixes crossing
  quantiles, and prints test metrics and examples.

## User preferences
- **Bare minimum code** for the MVP; short files; don't add features that weren't asked for.
- **Functions of 15 lines or fewer**, DRY, small modular functions. Check function lengths after edits.
- **Keep `main.py` simple.** One adapter instance, no method chaining, no extra adapter objects.
- **All filtering (including model choice and row cap) goes into the single adapter call.** The cap applies after filtering.
- **Train/test split lives in `main.py` as a plain 2-line percentage split** on the adapter's DataFrame.
- **The chat model is a filter, not a feature** - train on one model at a time so response styles don't mix.
- **First turn only for now; English only.**
- **Bucket edges are hardcoded, human-readable heuristics**, not computed from the data.
- Code is fully commented (docstrings on every function, inline comments on non-obvious lines).
- Likes to understand the reasoning: explain the ML concepts behind choices when asked, and give a recommendation.
- Plans to try many approaches eventually, but wants the simple version first.

## Work done so far
1. Built the adapter: all 6 shards, filter pushdown while reading (only matching rows reach memory), pair explosion,
   token counts, fixed buckets.
2. Classifier: TF-IDF (1-2 grams) + multinomial logistic regression with balanced class weights.
   - gpt-4 (5k rows): 53.4% accuracy, macro-F1 0.49, 85.8% within one bucket.
   - claude-1 (all rows, C=3 tuned by 5-fold CV): 62.8% accuracy, macro-F1 0.44, 84.0% within one bucket,
     vs. baselines majority 42.1% and prompt-length-only 33.7%.
3. Replaced the classifier with **quantile regression** (current code). s.
   9.4% of test prompts had crossing quantiles before sorting.

## Design choices
- **One row per prompt/response pair**, keeping `turn_index` and `num_turns`, so multi-turn work can reuse the adapter.
- **Token counts use tiktoken `cl100k_base`** for every model - a consistent approximation, not each model's real tokenizer.
- **Filters on raw parquet columns (model, language, redacted) are pushed into `pyarrow.read_table`**; filters on
  derived columns (`turn_index`) run after exploding. This is what makes loading all 1M conversations feasible.
- **Empty responses are dropped** (errors, not real short answers). **The adapter always shuffles** (seed 0), which makes
  the positional split in `main.py` random and repeatable.
- **Buckets** (1 token ≈ 0.75 words): ≤50 short answer · 51-150 paragraph · 151-300 few paragraphs · 301-600 page ·
  601+ multi-page. They suit gpt-4, but claude-1 has almost nothing in the 601+ bucket.
- **Why quantile regression:** the scheduler needs ordering (Spearman) plus a usable upper bound for reserving
  capacity; averages from Ridge-style regression compress toward the mean. p50 is the middle estimate, p90 the upper bound.
- **Model details:** `QuantileRegressor(solver="highs")` on TF-IDF features, target `log1p(tokens)` via
  `TransformedTargetRegressor` (quantiles survive the log/exp round trip). L1 `alpha` tuned per quantile with 3-fold CV on
  training rows only, scored by pinball loss. Grid is `[1e-4, 1e-3, 1e-2]`: 1e-4 always wins, but 1e-5 and below take
  more than 4 minutes per fit, so the grid can't go lower. Only ~545 of ~68k features have non-zero weights.
- **Each quantile is an independent model**, so predictions can cross; `uncross` sorts each prompt's predictions.
- **Baselines are always reported:** constant (training quantile) and prompt-length-only quantile regression.
- **Metrics:** coverage (should match the quantile), pinball loss, median absolute error, Spearman rank correlation;
  bucket accuracy and within-one-bucket accuracy for p50 only, to compare against the old classifier.

## Known limitations
- Each prompt has a single recorded response, and sampling noise is large, so no model can be very precise.
- The p25-p75 range is too narrow (p25 runs high).
- Response caps in the dataset mean long outputs are underrepresented.
- Redacted conversations are included.

## Future things to explore
- **Calibration:** widen or shift quantiles using a held-out part of the training data (e.g., conformal quantile regression)
  so p25-p75 really covers 50%.
- **Faster / non-linear quantile models:** LightGBM quantile objective (removes the alpha speed limit, often crosses less).
- **Better text features:** sentence embeddings (e.g., `all-MiniLM-L6-v2`) + linear model; fine-tuning a small transformer
  (DistilBERT/ModernBERT) - claude-1 has the most data for this.
- **Hand-made features:** requested length parsed from the prompt ("1000 words", "list 100").
- **Multi-turn:** include previous user and assistant messages as features (they determine what is already in the KV cache).
- **Other target models:** gpt-4, gpt-3.5-turbo, claude-instant-1, claude-2; decide whether buckets should be per model.
- **Other formulations:** Ridge/Poisson regression on log tokens, ordinal regression, ranking losses (only ordering matters
  for shortest-first), probability-weighted expected length from the classifier.
- **Data cleaning:** filter redacted rows; use model-specific tokenizers for exact counts.
- **Toward the real use case:** simulate a scheduler (e.g., average wait time under shortest-predicted-first vs FIFO),
  measure prediction latency, save the trained model and expose a predict function.
