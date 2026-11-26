"""ARF must generate the label jointly with the features."""

import numpy as np
import pandas as pd

from katabatic.models.arf.models import ARFModel


def _training_table(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n)
    hours = rng.normal(40, 10, n)
    odds = (age - 40) / 8 + (hours - 40) / 10
    label = np.where(rng.random(n) < 1 / (1 + np.exp(-odds)), "high", "low")
    return pd.DataFrame({"age": age, "hours": hours, "class": label})


def test_synthetic_label_depends_on_features(tmp_path):
    df = _training_table()
    df.to_csv(tmp_path / "train_full.csv", index=False)

    model = ARFModel(verbose=False)
    model.train(str(tmp_path), synthetic_dir=str(tmp_path / "synth"))
    synthetic = model.sample(2000)

    assert synthetic.columns.tolist() == df.columns.tolist()
    real_gap = df.groupby("class")["age"].mean().diff().abs().iloc[-1]
    synth_gap = synthetic.groupby("class")["age"].mean().diff().abs().iloc[-1]
    # An independently sampled label gives a gap close to zero.
    assert synth_gap > 0.6 * real_gap


def _categorical_table(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    married = rng.random(n) < 0.5
    label = np.where(rng.random(n) < np.where(married, 0.45, 0.05), "high", "low")
    status = np.where(married, "married", "single")
    return pd.DataFrame(
        {
            "status": status,
            "region": rng.choice(list("abcd"), n),
            "age": rng.integers(18, 70, n),
            "class": label,
        }
    )


def test_label_rate_per_category_is_preserved(tmp_path):
    df = _categorical_table()
    df.to_csv(tmp_path / "train_full.csv", index=False)

    model = ARFModel(verbose=False)
    model.train(str(tmp_path), synthetic_dir=str(tmp_path / "synth"))
    synthetic = model.sample(3000)

    def rate(frame):
        return (frame["class"] == "high").groupby(frame["status"]).mean()

    real, synth = rate(df), rate(synthetic)
    real_gap = real["married"] - real["single"]
    synth_gap = synth["married"] - synth["single"]
    assert real_gap > 0.3
    # Flat rates across categories (the old behaviour) give a gap near zero.
    assert synth_gap > 0.6 * real_gap
