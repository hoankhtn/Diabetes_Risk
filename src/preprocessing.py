"""Explicit, non-mutating numeric cleaning with row audits after each step."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from .utils import track_rows


def _columns(df: pd.DataFrame, columns: str | Sequence[str]) -> list[str]:
    """Require explicit columns so IDs/labels are never selected implicitly."""
    names = [columns] if isinstance(columns, str) else list(columns)
    if not names or len(names) != len(set(names)):
        raise ValueError("columns must be non-empty and contain no duplicates.")
    if not df.columns.is_unique:
        raise ValueError("DataFrame has duplicate column names.")
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return names


def _numeric_values(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    for name in names:
        dtype = df[name].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise TypeError(f"{name!r} must be real numeric; use clean_numeric_columns first.")
    return df[names].astype(float)


def replace_placeholders(
    df: pd.DataFrame,
    columns: str | Sequence[str],
    placeholders: Sequence[float] = (-1, 999),
    *,
    step_name: str = "Replace numeric placeholders",
) -> pd.DataFrame:
    """Replace confirmed numeric placeholder codes with NaN in selected columns.

    No rows are dropped. String tokens are handled by clean_numeric_columns,
    which converts to numbers first. Caller must confirm that the chosen codes
    represent missing data in each selected column; IDs should be excluded.
    """
    names = _columns(df, columns)
    _numeric_values(df, names)
    mask = df[names].isin(list(placeholders))
    result = df.copy()
    for name in names:
        result[name] = result[name].mask(mask[name])
    print(f"[{step_name}] replaced cells by column: {mask.sum().to_dict()}.")
    track_rows(df, result, step_name, reason="Placeholder cells became missing; no rows dropped.")
    return result


def detect_outliers(
    df: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Return per-cell Boolean flags, preserving the input's index and columns.

    IQR uses Q1-k*IQR and Q3+k*IQR (linear quantiles, fences rounded to 10
    decimals); z-score uses abs((x-mean)/std) > threshold with ddof=0.
    These conventions match notebook 01. Bounds are inclusive (only values
    strictly outside are flagged). Missing values, zero IQR/variance columns
    and all-missing columns are not flagged. Infinite values must be cleaned
    first. Statistics are computed once on the supplied frame, per column.
    """
    names = _columns(df, columns)
    if method not in {"iqr", "zscore"}:
        raise ValueError("method must be 'iqr' or 'zscore'.")
    threshold = iqr_multiplier if method == "iqr" else z_threshold
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Outlier threshold must be finite and positive.")
    values = _numeric_values(df, names)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Infinite values found; use clean_numeric_columns first.")
    if method == "iqr":
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        spread = q3 - q1
        lower = (q1 - iqr_multiplier * spread).round(10)
        upper = (q3 + iqr_multiplier * spread).round(10)
        flags = (values.lt(lower) | values.gt(upper)) & spread.gt(0)
    else:
        spread = values.std(ddof=0).replace(0, np.nan)
        flags = ((values - values.mean()) / spread).abs().gt(z_threshold)
    return flags.fillna(False).astype(bool)


def filter_outliers(
    df: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    step_name: str = "Filter numeric outliers",
) -> pd.DataFrame:
    """Drop rows flagged in ANY selected column and report distinct row loss.

    Missing values are retained. Degenerate columns follow detect_outliers.
    This is an explicit sensitivity-analysis option, not an automatic claim
    that extreme values are data errors. The input is never modified.
    """
    flags = detect_outliers(
        df, columns, method=method,
        iqr_multiplier=iqr_multiplier, z_threshold=z_threshold,
    )
    result = df.loc[~flags.any(axis=1)].copy()
    print(f"[{step_name}] {method} flags by column: {flags.sum().to_dict()}.")
    track_rows(
        df, result, step_name,
        reason=f"Removed {int(flags.any(axis=1).sum()):,} distinct rows flagged by {method} "
               "in at least one selected column; missing values retained.",
    )
    return result


def clean_numeric_columns(
    df: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    placeholders: Sequence[float] = (-1, 999),
    drop_missing: bool = False,
    outlier_method: str | None = None,
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Coerce numbers, replace placeholders, optionally drop missing/outliers.

    Invalid numeric strings and +/-infinity become missing; numeric strings
    such as '999' are subsequently recognized as placeholders. Each step
    prints cell counts and/or a row-count audit. Defaults retain every row.
    Other columns and the index are preserved; the input is not modified.
    For ML, fit imputation/scaling/outlier thresholds on TRAIN only rather
    than applying this descriptive filter to the full dataset before splitting.
    """
    names = _columns(df, columns)
    if outlier_method not in {None, "iqr", "zscore"}:
        raise ValueError("outlier_method must be None, 'iqr', or 'zscore'.")
    converted = df.copy()
    newly_missing = {}
    for name in names:
        converted[name] = pd.to_numeric(df[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan,
        )
        newly_missing[name] = int((df[name].notna() & converted[name].isna()).sum())
    print(f"[Convert numeric columns] invalid/non-finite cells made missing: {newly_missing}.")
    track_rows(df, converted, "Convert numeric columns", reason="Changed cells only; no rows dropped.")
    result = replace_placeholders(converted, names, placeholders)
    if drop_missing:
        before = result
        result = before.dropna(subset=names).copy()
        track_rows(before, result, "Drop missing numeric values",
                   reason=f"Removed rows missing at least one of {names}.")
    if outlier_method is not None:
        result = filter_outliers(
            result, names, method=outlier_method,
            iqr_multiplier=iqr_multiplier, z_threshold=z_threshold,
        )
    return result
