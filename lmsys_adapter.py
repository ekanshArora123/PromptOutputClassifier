"""Adapter over the LMSYS-Chat-1M parquet shards.

The raw dataset has one row per conversation, with the whole chat stored as a list of
{"role", "content"} messages. This adapter turns that into one row per (user prompt,
assistant response) pair, adds token counts and a response-length bucket, and exposes a
few simple filter/sort/sample helpers that return plain pandas DataFrames.
"""
import pandas as pd
import pyarrow.parquet as pq
import tiktoken

# Columns read from the parquet files. `openai_moderation` is skipped because it is large and unused.
COLUMNS = ["conversation_id", "model", "language", "turn", "redacted", "conversation"]

# OpenAI's GPT-4-era tokenizer. Each chat model in the dataset uses its own tokenizer, so these
# counts are a consistent approximation rather than the exact number of tokens each model generated.
ENCODER = tiktoken.get_encoding("cl100k_base")

INF = float("inf")

# Fixed, human-readable response-length buckets: (max response tokens, label).
# Rule of thumb: 1 token ~ 0.75 English words. Upper edges are inclusive, so exactly 50 tokens is bucket 0.
BUCKETS = [
    (50, "short answer (a sentence or two)"),
    (150, "one paragraph"),
    (300, "a few paragraphs"),
    (600, "about a page"),
    (INF, "long form (multi-page)"),
]
# Bin edges for pd.cut: (-inf, 50], (50, 150], (150, 300], (300, 600], (600, inf)
BUCKET_EDGES = [-INF] + [upper for upper, _ in BUCKETS]


class LmsysAdapter:
    """Loads the dataset once into `self.df`; the helper methods return filtered, sorted or sampled copies of it.

    Columns of `self.df`:
        conversation_id, model, language, redacted - copied from the raw conversation
        num_turns - number of user turns in the whole conversation (the raw `turn` column)
        turn_index - 0 for a conversation's first prompt/response pair, 1 for the second, ...
        prompt, response - the user message and the assistant reply that followed it
        prompt_tokens, response_tokens - token counts from ENCODER
        bucket - 0-based index into BUCKETS for response_tokens
    """

    def __init__(self, data_dir, max_rows=None, seed=0, **equals):
        """Load pairs matching `equals` (e.g. model="gpt-4", turn_index=0), then drop empty replies, shuffle, cap, bucket.

        Filters on raw parquet columns (model, language, redacted) run while reading, so non-matching
        conversations never reach memory. Filters on pair columns (turn_index) run after exploding.
        """
        raw_filters = {col: value for col, value in equals.items() if col in COLUMNS}
        self.df = _read_pairs(data_dir, raw_filters)
        self.df = self.filter_eq(**{col: value for col, value in equals.items() if col not in raw_filters})
        self.df = self.filter_range("response_tokens", low=1)  # empty replies are errors, not short answers
        self.df = self.sample(max_rows, seed)  # shuffled, so callers can split train/test by position
        self.df = self.df.assign(bucket=pd.cut(self.df["response_tokens"], bins=BUCKET_EDGES, labels=False))

    def __len__(self):
        """Number of loaded prompt/response pairs."""
        return len(self.df)

    def filter_eq(self, **equals):
        """Rows where each column equals the value, or is one of the values when given a list.

        Example: filter_eq(model=["gpt-4", "claude-1"], turn_index=0)
        """
        mask = pd.Series(True, index=self.df.index)  # start by keeping every row
        for col, value in equals.items():
            mask &= self.df[col].isin(_as_list(value))  # every condition must hold
        return self.df[mask]

    def filter_range(self, col, low=-INF, high=INF):
        """Rows where `col` is between `low` and `high`, inclusive on both ends."""
        return self.df[self.df[col].between(low, high)]

    def sort(self, col, descending=False):
        """All rows ordered by `col`."""
        return self.df.sort_values(col, ascending=not descending)

    def sample(self, n=None, seed=0):
        """Rows in random order, at most `n` of them (all rows when `n` is None).

        The fixed `seed` makes the shuffle, and therefore any positional train/test split, repeatable.
        """
        n = len(self) if n is None else min(n, len(self))  # never ask for more rows than exist
        return self.df.sample(n=n, random_state=seed)


def _as_list(value):
    """Wrap a single filter value in a list so every filter can use an "is in" check."""
    return value if isinstance(value, list) else [value]


def _read_pairs(data_dir, equals):
    """Read every parquet shard in `data_dir`, keeping only conversations matching `equals`, and explode into pairs."""
    # pyarrow filter format: [(column, "in", [allowed values]), ...]; all conditions must hold.
    filters = [(col, "in", _as_list(value)) for col, value in equals.items()]
    table = pq.read_table(data_dir, columns=COLUMNS, filters=filters or None)  # None means no filtering
    return _to_pairs(table.to_pandas())


def _to_pairs(raw):
    """Explode each conversation into rows of a user prompt paired with the assistant reply that follows it."""
    # Messages alternate user, assistant, user, ...: message i (even) is a prompt and message i + 1 its reply.
    # Each output row keeps the conversation's metadata (model, language, ...) plus that turn's text.
    rows = [
        {**meta, "turn_index": i // 2, "prompt": msgs[i]["content"], "response": msgs[i + 1]["content"]}
        for meta, msgs in zip(raw.drop(columns="conversation").to_dict("records"), raw["conversation"])
        for i in range(0, len(msgs) - 1, 2)
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant"  # skip out-of-order messages
    ]
    pairs = pd.DataFrame(rows).rename(columns={"turn": "num_turns"})  # raw `turn` counts turns per conversation
    pairs["prompt_tokens"] = _count_tokens(pairs["prompt"])
    pairs["response_tokens"] = _count_tokens(pairs["response"])
    return pairs


def _count_tokens(texts):
    """Token count for each string, encoded across multiple threads by tiktoken."""
    # disallowed_special=() treats text like "<|endoftext|>" inside a message as plain text instead of raising.
    return [len(tokens) for tokens in ENCODER.encode_batch(texts.tolist(), disallowed_special=())]
