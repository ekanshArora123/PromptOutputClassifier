"""Train and evaluate quantile regression models that predict how many tokens a chat model will generate.

Pipeline:
    1. Load first-turn English prompt/response pairs for one chat model through LmsysAdapter.
    2. Split them by position into train and test (the adapter has already shuffled them).
    3. For each quantile, fit a constant baseline, a prompt-length baseline and a TF-IDF model
       whose L1 strength is tuned by cross-validation on the training rows only.
    4. Sort each prompt's quantile predictions so they never cross, then score every model on the test rows.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from lmsys_adapter import BUCKET_EDGES, LmsysAdapter

DATA_DIR = Path(__file__).parent / "data"  # folder holding the 6 parquet shards
TARGET_MODEL = "claude-1"  # train on one chat model's responses so different models' lengths don't mix
MAX_ROWS = None  # total cap (train + test); None uses every matching row
TRAIN_FRACTION = 0.8  # first 80% of the shuffled rows train, the last 20% test
QUANTILES = [0.25, 0.5, 0.75, 0.9]  # p25-p75 = middle half of likely lengths, p50 = middle estimate, p90 = upper bound
ALPHA_GRID = [1e-4, 1e-3, 1e-2]  # L1 strengths to try (larger = stronger); below 1e-4 a single fit takes minutes
# GridSearchCV names nested settings by path: TransformedTargetRegressor -> pipeline -> QuantileRegressor -> alpha.
ALPHA_PARAM = "regressor__quantileregressor__alpha"


def load_splits():
    """Load the filtered pairs and split them into (train, test) DataFrames by position."""
    data = LmsysAdapter(DATA_DIR, max_rows=MAX_ROWS, model=TARGET_MODEL, language="English", turn_index=0)
    cut = int(len(data) * TRAIN_FRACTION)
    return data.df.iloc[:cut], data.df.iloc[cut:]  # rows are already shuffled, so this is a random split


def _quantile_model(transformer, column, quantile):
    """Linear quantile regression on log(1 + tokens) from one feature column; predictions come back in tokens.

    `transformer` turns `column` of the pairs DataFrame into numeric features. QuantileRegressor minimizes
    pinball loss + alpha * sum(|weights|), solved exactly as a linear program by the HiGHS solver.
    """
    features = make_column_transformer((transformer, column))  # pick the column, then transform it
    regressor = make_pipeline(features, QuantileRegressor(quantile=quantile, alpha=ALPHA_GRID[0], solver="highs"))
    # Train on log(1 + tokens) so errors are relative; a quantile survives the log/exp round trip unchanged.
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def build_models(quantile):
    """Constant baseline, prompt-length baseline, and TF-IDF model with alpha tuned by cross-validation."""
    # TF-IDF over words and word pairs found in at least 2 training prompts; sublinear_tf uses 1 + log(count).
    tfidf = _quantile_model(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "prompt", quantile)
    # Score by pinball loss (here `alpha` means the quantile, not the L1 strength); negated because higher must be better.
    scorer = make_scorer(mean_pinball_loss, alpha=quantile, greater_is_better=False)
    return {
        "constant": DummyRegressor(strategy="quantile", quantile=quantile),  # training set's quantile for every prompt
        "prompt length only": _quantile_model(FunctionTransformer(np.log1p), ["prompt_tokens"], quantile),
        # Try each alpha with 3-fold cross-validation on the training rows, then refit the best one on all of them.
        "tf-idf": GridSearchCV(tfidf, {ALPHA_PARAM: ALPHA_GRID}, scoring=scorer, cv=3, n_jobs=-1),
    }


def fit_predict(model, train_df, test_df):
    """Fit `model` on the training rows and return its token predictions for the test rows."""
    model.fit(train_df, train_df["response_tokens"])  # each model selects its own feature column
    return model.predict(test_df)


def scores(y_true, y_pred, quantile):
    """Test metrics for one model's predictions at one quantile."""
    result = {
        "coverage": np.mean(y_true <= y_pred),  # share of responses at or under the prediction; target = quantile
        "pinball loss": mean_pinball_loss(y_true, y_pred, alpha=quantile),  # training objective in tokens; lower is better
        "median abs error": np.median(np.abs(y_true - y_pred)),  # typical miss in tokens
        # Rank agreement between predicted and actual lengths, which is what shortest-first scheduling needs.
        # Undefined for a constant prediction, so report NaN instead of triggering a warning.
        "spearman": pd.Series(y_pred).corr(pd.Series(y_true), method="spearman") if np.ptp(y_pred) else np.nan,
    }
    # Bucket metrics only suit the middle estimate, which is what compares to the old classifier.
    return result | (bucket_scores(y_true, y_pred) if quantile == 0.5 else {})


def bucket_scores(y_true, y_pred):
    """Bucket the median predictions so they can be compared with the old classifier."""
    # Map actual and predicted token counts to bucket indices 0-4 using the adapter's fixed edges.
    true_bucket, pred_bucket = (pd.cut(values, bins=BUCKET_EDGES, labels=False) for values in (y_true, y_pred))
    return {
        "bucket accuracy": np.mean(true_bucket == pred_bucket),
        "within one bucket": np.mean(np.abs(true_bucket - pred_bucket) <= 1),  # off by at most one bucket
    }


def collect_predictions(train_df, test_df):
    """Fit every model at every quantile; returns {model name: {quantile: test predictions}} and tuned alphas."""
    predictions, alphas = defaultdict(dict), {}  # defaultdict creates each model's inner dict on first use
    for quantile in QUANTILES:
        models = build_models(quantile)  # fresh, unfitted models for this quantile
        for name, model in models.items():
            predictions[name][quantile] = fit_predict(model, train_df, test_df)
        alphas[quantile] = models["tf-idf"].best_params_[ALPHA_PARAM]  # alpha chosen by cross-validation
    return predictions, alphas


def uncross(predictions):
    """Sort each prompt's predictions across quantiles so a higher quantile is never below a lower one.

    Each quantile model is trained independently, so for some prompts p75 can come out below p50.
    Sorting uses only the predictions, never the actual response lengths.
    """
    print()
    for name, by_quantile in predictions.items():
        stacked = np.column_stack([by_quantile[q] for q in QUANTILES])  # shape: (test prompts, quantiles)
        crossed = np.any(np.diff(stacked, axis=1) < 0, axis=1).mean()  # share of prompts with any out-of-order pair
        print(f"{name}: {crossed:.1%} of test prompts had crossing quantiles (fixed by sorting)")
        predictions[name] = dict(zip(QUANTILES, np.sort(stacked, axis=1).T))  # sort each row, split back per quantile
    return predictions


def print_scores(y_true, predictions, alphas):
    """Print one table per quantile comparing every model's test metrics."""
    for quantile in QUANTILES:
        print()
        print(f"=== quantile {quantile} (tf-idf best alpha: {alphas[quantile]}) ===")
        table = {name: scores(y_true, by_quantile[quantile], quantile) for name, by_quantile in predictions.items()}
        print(pd.DataFrame(table).T.round(3).to_string())  # transpose so each model is a row


def print_range(y_true, by_quantile, low=0.25, high=0.75):
    """Print how often actual lengths land inside the low-high quantile range, and how wide that range is."""
    inside = np.mean((by_quantile[low] <= y_true) & (y_true <= by_quantile[high]))  # p25-p75 should hold 50%
    width = np.median(by_quantile[high] - by_quantile[low])  # typical width in tokens; narrower = more confident
    print()
    print(f"tf-idf p{low * 100:g}-p{high * 100:g} range: {inside:.1%} of test responses fall inside "
          f"(target {high - low:.0%}), median width {width:.0f} tokens")


def print_examples(test_df, predictions, n=8):
    """Print the first `n` test prompts with their actual length and every quantile prediction."""
    columns = {f"p{round(q * 100)}": pred[:n].round().astype(int) for q, pred in predictions.items()}  # {"p25": ...}
    examples = test_df.head(n).assign(**columns)[["response_tokens", *columns, "prompt"]]
    print()
    print("=== example test predictions (tokens) ===")
    print(examples.to_string(max_colwidth=60))


def main():
    """Load data, train all models, fix crossing quantiles and print the test results."""
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles otherwise fail on non-English prompt text
    train_df, test_df = load_splits()
    print(f"{TARGET_MODEL}: {len(train_df)} train rows, {len(test_df)} test rows")
    predictions, alphas = collect_predictions(train_df, test_df)
    predictions = uncross(predictions)
    y_true = test_df["response_tokens"].to_numpy()  # actual lengths, used only for scoring
    print_scores(y_true, predictions, alphas)
    print_range(y_true, predictions["tf-idf"])
    print_examples(test_df, predictions["tf-idf"])


if __name__ == "__main__":
    main()
