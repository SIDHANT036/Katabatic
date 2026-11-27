"""CTGAN must learn how the label depends on the other columns.

Before, real rows were paired with a multi-hot condition while generated rows
used a single one-hot condition, so the discriminator could separate them by
the condition alone and the generator never learned cross-column structure.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from katabatic.models.ctgan.models import CTGANModel  # noqa: E402
from katabatic.models.ctgan.utils import (  # noqa: E402
    ConditionSampler,
    build_conditioning,
    decode_batch,
    encode_df,
    fit_transformers,
    infer_schema,
)


def _frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    married = rng.random(n) < 0.5
    high = rng.random(n) < np.where(married, 0.6, 0.05)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "status": np.where(married, "married", "single"),
            "region": rng.choice(list("abc"), n),
            "class": np.where(high, "high", "low"),
        }
    )


def test_declared_column_types_override_the_heuristic():
    df = _frame()
    # age is an integer with low relative cardinality, so it is guessed categorical.
    guessed = {c.name: c.kind for c in infer_schema(df)}
    declared = {c.name: c.kind for c in infer_schema(df, continuous_cols=["age"])}
    assert guessed["age"] == "categorical"
    assert declared["age"] == "continuous"
    assert declared["class"] == "categorical"  # undeclared label still inferred


def test_condition_sampler_matches_cond_and_rows():
    df = _frame()
    schema = infer_schema(df, continuous_cols=["age"])
    fit_transformers(df, schema)
    _, cond_blocks = build_conditioning(df, schema)
    probs = {
        c.name: df[c.name]
        .astype(str)
        .value_counts(normalize=True)
        .reindex(c.categories)
        .to_numpy()
        for c in schema
        if c.kind == "categorical"
    }
    sampler = ConditionSampler(df, schema, cond_blocks, probs)
    cond, rows, col_idx, cat_idx = sampler.sample(400, np.random.default_rng(0))

    assert (cond.sum(axis=1) == 1).all()  # a single one-hot, like the fake side
    for j, col in enumerate(sampler.cols):
        picked = col_idx == j
        expected = np.asarray(col.categories)[cat_idx[picked]]
        assert (df[col.name].astype(str).to_numpy()[rows[picked]] == expected).all()


def test_encode_decode_roundtrip_keeps_continuous_values():
    df = _frame()
    schema = infer_schema(df, continuous_cols=["age"])
    fit_transformers(df, schema)
    enc, _, _ = encode_df(df, schema)
    assert enc[:, 0].min() >= -1.0 and enc[:, 0].max() <= 1.0  # fits tanh
    decoded = decode_batch(enc, schema)
    assert np.abs(decoded["age"].to_numpy() - df["age"].to_numpy()).max() < 3
    assert (decoded["status"] == df["status"]).all()


def test_label_depends_on_features_after_sampling():
    df = _frame(n=2000)
    model = CTGANModel(
        epochs=150,
        batch_size=256,
        generator_hidden=(64, 64),
        discriminator_hidden=(64, 64),
        seed=0,
    )
    model.fit(df, categorical_cols=["status", "region"], continuous_cols=["age"])
    synthetic = model.sample(3000)

    def gap(frame):
        rate = (frame["class"] == "high").groupby(frame["status"]).mean()
        return rate["married"] - rate["single"]

    assert gap(df) > 0.4
    # The old model produced a gap near zero here.
    assert gap(synthetic) > 0.5 * gap(df)
