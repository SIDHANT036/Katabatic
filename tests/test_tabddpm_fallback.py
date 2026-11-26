"""Regression tests for the local TabDDPM fallback diffusion.

The fallback used to compute its categorical loss on ``torch.round`` outputs
(zero gradient), so every categorical column collapsed to a single value and
the label carried no information about the features. These tests train a tiny
model on data with a strong label/feature dependence and check that categorical
columns keep their variety and the label dependence survives sampling.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from katabatic.experimental.models.tabddpm.models import (  # noqa: E402
    _TABDDPM_EXTERNAL,
    Tabddpm,
)

pytestmark = pytest.mark.skipif(
    _TABDDPM_EXTERNAL, reason="tests target the local fallback implementation"
)

POSITIVE = ">50K"


def _make_data(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    marital = np.where(
        y,
        rng.choice(["married", "single", "divorced"], n, p=[0.85, 0.1, 0.05]),
        rng.choice(["married", "single", "divorced"], n, p=[0.3, 0.55, 0.15]),
    )
    occupation = rng.choice(list("abcde"), n)
    age = np.where(y, rng.normal(0.4, 0.2, n), rng.normal(-0.2, 0.3, n)).clip(-1, 1)
    features = pd.DataFrame({"age": age, "marital": marital, "occupation": occupation})
    target = pd.Series(np.where(y, POSITIVE, "<=50K"))
    return features, target


@pytest.fixture(scope="module")
def trained():
    features, target = _make_data()
    model = Tabddpm()
    model.train(
        features,
        target,
        cat_cols=["marital", "occupation"],
        config=dict(
            steps=1000,
            num_timesteps=50,
            batch_size=256,
            d_layers=(128, 128),
            eval_batches=2,
            seed=0,
        ),
    )
    return features, target, model


def _positive_rate(frame, label_col, by):
    return (frame[label_col] == POSITIVE).groupby(frame[by]).mean()


def test_sample_has_schema_and_no_nan(trained):
    features, _, model = trained
    synthetic = model.sample(500)
    assert set(synthetic.columns) == set(features.columns) | {"label"}
    assert not synthetic.isna().any().any()
    assert set(synthetic["marital"]) <= set(features["marital"])


def test_categorical_columns_do_not_collapse(trained):
    _, _, model = trained
    synthetic = model.sample(1000)
    # The broken fallback emitted exactly one value per categorical column.
    assert synthetic["marital"].nunique() >= 2
    assert synthetic["occupation"].nunique() >= 4


def test_label_depends_on_features_after_sampling(trained):
    features, target, model = trained
    real = features.assign(label=target.to_numpy())
    synthetic = model.sample(3000)
    real_rate = _positive_rate(real, "label", "marital")
    synth_rate = _positive_rate(synthetic, "label", "marital")
    real_gap = real_rate["married"] - real_rate["single"]
    synth_gap = synth_rate["married"] - synth_rate["single"]
    assert real_gap > 0.3
    assert synth_gap > 0.5 * real_gap


def test_categorical_loss_has_gradient():
    import torch

    from katabatic.experimental.models.tabddpm.utils import (
        GaussianMultinomialDiffusion,
        MLPDiffusion,
    )

    denoiser = MLPDiffusion(
        d_in=1 + 3, num_classes=2, is_y_cond=True, rtdl_params={"d_layers": [16]}
    )
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array([3]),
        num_numerical_features=1,
        denoise_fn=denoiser,
        num_timesteps=10,
    )
    xb = torch.tensor([[0.1, 0.0], [0.2, 1.0], [-0.3, 2.0], [0.0, 1.0]])
    loss_multi, _ = diffusion.mixed_loss(xb, {"y": torch.tensor([0, 1, 0, 1])})
    loss_multi.backward()
    grads = [p.grad for p in denoiser.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
