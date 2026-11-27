from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


def infer_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Infer categorical columns using dtype heuristics and cardinality.

    Rules:
    - object or category dtypes are categorical
    - integer columns with very low absolute cardinality
    (<=20 distinct values) or low relative cardinality (<5% of rows)
      are treated as categorical
    """
    cat_cols: list[str] = []
    n_rows = max(len(df), 1)

    for col in df.columns:
        s = df[col]
        dt = str(s.dtype)
        if dt == "object" or dt.startswith("category"):
            cat_cols.append(col)
            continue
        if dt.startswith("int"):
            nunique = s.nunique(dropna=True)
            if nunique <= 20 or nunique / n_rows < 0.05:
                cat_cols.append(col)

    return cat_cols


@dataclass
class ColumnMeta:
    name: str
    kind: str  # 'continuous' | 'categorical'
    categories: list[str] | None = None
    qt: QuantileTransformer | None = None


def infer_schema(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    continuous_cols: list[str] | None = None,
) -> list[ColumnMeta]:
    """Build the column schema.

    Columns named in ``categorical_cols`` or ``continuous_cols`` use that type;
    every other column (for example the label) falls back to the dtype and
    cardinality heuristic.
    """
    schema: list[ColumnMeta] = []
    declared_cat = set(categorical_cols or [])
    declared_cont = set(continuous_cols or [])
    cat_cols = (set(infer_categorical_columns(df)) - declared_cont) | (
        declared_cat & set(df.columns)
    )
    for c in df.columns:
        if c in cat_cols:
            cats = sorted([str(v) for v in df[c].dropna().unique().tolist()])
            schema.append(ColumnMeta(name=c, kind="categorical", categories=cats))
        else:
            schema.append(ColumnMeta(name=c, kind="continuous"))
    return schema


# Quantile-normal values are clipped to +/-CONT_SCALE and divided by it so they
# fit the generator's tanh output range of (-1, 1).
CONT_SCALE = 3.0


def fit_transformers(df: pd.DataFrame, schema: list[ColumnMeta]) -> None:
    for col in schema:
        if col.kind == "continuous":
            qt = QuantileTransformer(
                output_distribution="normal", n_quantiles=min(1000, max(len(df), 10))
            )
            vals = df[[col.name]].astype(float)
            qt.fit(vals)
            col.qt = qt
        else:
            # categories already stored (sorted strings)
            pass


def encode_df(
    df: pd.DataFrame, schema: list[ColumnMeta]
) -> tuple[np.ndarray, dict[str, tuple[int, int]], list[str]]:
    """Encode dataframe to a continuous + one-hot representation.

    Returns: (encoded_array, cat_blocks, output_order)
    - cat_blocks: mapping col_name -> (start, end) indices of its one-hot block
    - output_order: list of original column names in order encountered
    """
    arrays: list[np.ndarray] = []
    cat_blocks: dict[str, tuple[int, int]] = {}
    output_order: list[str] = []

    for col in schema:
        output_order.append(col.name)
        if col.kind == "continuous":
            vals = df[[col.name]].astype(float).values
            if col.qt is not None:
                vals = np.clip(col.qt.transform(vals), -CONT_SCALE, CONT_SCALE)
                vals = vals / CONT_SCALE
            arrays.append(vals.astype(np.float32))
        else:
            cats = col.categories or []
            # Build mapping str(value)->index
            cat_to_idx = {v: i for i, v in enumerate(cats)}
            idxs = df[col.name].astype(str).map(lambda v: cat_to_idx.get(v, -1)).values
            one_hot = np.zeros((len(df), len(cats)), dtype=np.float32)
            valid = idxs >= 0
            one_hot[np.where(valid)[0], idxs[valid]] = 1.0
            start = int(sum(a.shape[1] for a in arrays))
            arrays.append(one_hot)
            end = int(sum(a.shape[1] for a in arrays))
            cat_blocks[col.name] = (start, end)

    enc = (
        np.concatenate(arrays, axis=1)
        if arrays
        else np.zeros((len(df), 0), dtype=np.float32)
    )
    return enc, cat_blocks, output_order


def decode_batch(enc: np.ndarray, schema: list[ColumnMeta]) -> pd.DataFrame:
    """Decode generator output back to the original columns (vectorised)."""
    data: dict[str, object] = {}
    ptr = 0
    for col in schema:
        if col.kind == "continuous":
            vals = enc[:, ptr].astype(float)
            ptr += 1
            if col.qt is not None:
                vals = col.qt.inverse_transform((vals * CONT_SCALE).reshape(-1, 1))
                vals = vals.ravel()
            data[col.name] = vals
        else:
            cats = col.categories or []
            size = len(cats)
            if size == 0:
                data[col.name] = np.full(enc.shape[0], None, dtype=object)
            else:
                idx = np.argmax(enc[:, ptr : ptr + size], axis=1)
                data[col.name] = np.asarray(cats, dtype=object)[idx]
            ptr += size
    return pd.DataFrame(data, columns=[c.name for c in schema])


def build_conditioning(
    df: pd.DataFrame, schema: list[ColumnMeta]
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Build a full one-hot condition space across all categorical columns.

    Returns: (cond_matrix for each row based on actual categories, cond_blocks mapping)
    """
    arrays: list[np.ndarray] = []
    cond_blocks: dict[str, tuple[int, int]] = {}
    for col in schema:
        if col.kind == "categorical":
            cats = col.categories or []
            cat_to_idx = {v: i for i, v in enumerate(cats)}
            idxs = df[col.name].astype(str).map(lambda v: cat_to_idx.get(v, -1)).values
            one_hot = np.zeros((len(df), len(cats)), dtype=np.float32)
            valid = idxs >= 0
            one_hot[np.where(valid)[0], idxs[valid]] = 1.0
            start = int(sum(a.shape[1] for a in arrays))
            arrays.append(one_hot)
            end = int(sum(a.shape[1] for a in arrays))
            cond_blocks[col.name] = (start, end)
    cond = (
        np.concatenate(arrays, axis=1)
        if arrays
        else np.zeros((len(df), 0), dtype=np.float32)
    )
    return cond, cond_blocks


def sample_conditions(
    n: int,
    schema: list[ColumnMeta],
    cond_blocks: dict[str, tuple[int, int]],
    empirical_probs: dict[str, np.ndarray] | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Sample condition vectors by choosing one categorical column uniformly and a category per its distribution."""
    total_dim = 0
    spans: list[tuple[str, int]] = []
    for col in schema:
        if col.kind == "categorical":
            dim = len(col.categories or [])
            spans.append((col.name, dim))
            total_dim += dim
    if total_dim == 0:
        return np.zeros((n, 0), dtype=np.float32)

    conds = np.zeros((n, total_dim), dtype=np.float32)
    for i in range(n):
        # choose a categorical column uniformly
        j = np.random.randint(len(spans))
        name, dim = spans[j]
        start, end = cond_blocks[name]
        if empirical_probs and name in empirical_probs:
            p = empirical_probs[name]
            k = int(np.random.choice(len(p), p=p))
        else:
            k = int(np.random.randint(dim))
        conds[i, start + k] = 1.0
    return conds.astype(np.float32)


class ConditionSampler:
    """Training-by-sampling: a condition and real rows that satisfy it.

    Each draw picks one categorical column uniformly, a category from that
    column's empirical distribution, and a real row holding that category. The
    real batch and the generator therefore see the same single one-hot
    condition, so the discriminator cannot tell them apart by its shape.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: list[ColumnMeta],
        cond_blocks: dict[str, tuple[int, int]],
        empirical_probs: dict[str, np.ndarray],
    ) -> None:
        self.cols = [c for c in schema if c.kind == "categorical" and c.categories]
        self.cond_blocks = cond_blocks
        self.cond_dim = int(sum(len(c.categories) for c in self.cols))
        self.probs = []
        self.order = []
        self.starts = []
        self.counts = []
        for col in self.cols:
            cat_to_idx = {v: i for i, v in enumerate(col.categories)}
            codes = df[col.name].astype(str).map(lambda v: cat_to_idx.get(v, -1))
            codes = codes.to_numpy()
            order = np.argsort(codes, kind="stable")
            k = len(col.categories)
            counts = np.bincount(codes[codes >= 0], minlength=k)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]]) + int(
                (codes < 0).sum()
            )
            p = np.asarray(empirical_probs[col.name], dtype=float) * (counts > 0)
            self.probs.append(p / p.sum())
            self.order.append(order)
            self.starts.append(starts)
            self.counts.append(counts)

    def sample(self, n: int, rng: np.random.Generator):
        """Return (cond, row_idx, col_idx, cat_idx) for ``n`` draws."""
        col_idx = rng.integers(len(self.cols), size=n)
        cat_idx = np.zeros(n, dtype=int)
        row_idx = np.zeros(n, dtype=int)
        cond = np.zeros((n, self.cond_dim), dtype=np.float32)
        for j, col in enumerate(self.cols):
            rows = np.where(col_idx == j)[0]
            if len(rows) == 0:
                continue
            cats = rng.choice(len(self.probs[j]), size=len(rows), p=self.probs[j])
            offset = np.floor(rng.random(len(rows)) * self.counts[j][cats]).astype(int)
            row_idx[rows] = self.order[j][self.starts[j][cats] + offset]
            cat_idx[rows] = cats
            start, _ = self.cond_blocks[col.name]
            cond[rows, start + cats] = 1.0
        return cond, row_idx, col_idx, cat_idx
