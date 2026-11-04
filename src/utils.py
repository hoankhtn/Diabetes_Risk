"""Row-count and merge audits for section 3.4 and D1."""

from collections.abc import Sequence
from typing import Any

import pandas as pd


def track_rows(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    step_name: str,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Print and return a row-count audit without modifying either frame.

    ``change`` is after minus before (a net change, not a count of deleted
    records). Percentage change is undefined (None) for an empty input.
    Supply ``reason`` or a descriptive step name to explain the operation;
    counts alone cannot establish why rows changed.
    """
    before, after = len(df_before), len(df_after)
    change = after - before
    change_pct = 100 * change / before if before else None
    report = {
        "step_name": step_name,
        "rows_before": before,
        "rows_after": after,
        "change": change,
        "change_pct": change_pct,
        "reason": reason,
    }
    pct_text = f"{change_pct:+.2f}%" if change_pct is not None else "N/A"
    print(f"[{step_name}] rows: {before:,} -> {after:,} "
          f"(change {change:+,}; {pct_text})."
          + (f" Reason: {reason}" if reason else ""))
    return report


def verify_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | Sequence[str],
    *,
    how: str = "left",
    validate: str = "many_to_one",
    step_name: str = "Merge",
    allow_null_keys: bool = False,
    suffixes: tuple[str, str] = ("_x", "_y"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge and return ``(merged, report)`` with coverage of BOTH inputs.

    Supports inner/left/right/outer joins on named columns, including composite
    keys. Unmatched key frames contain every distinct unmatched key, even when
    the requested join discards it. Unmatched row counts include duplicates in
    the original input. The coverage join uses distinct keys to avoid a
    many-to-many explosion during auditing.

    Cardinality defaults to many_to_one, suitable for a reference table. Pandas
    raises MergeError on violations. Null keys raise ValueError by default;
    opt-in uses pandas semantics, where null keys can match each other.
    """
    keys = [on] if isinstance(on, str) else list(on)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("on must contain at least one key, without duplicates.")
    if how not in {"left", "right", "inner", "outer"}:
        raise ValueError("how must be left, right, inner, or outer.")
    if validate not in {
        "one_to_one", "1:1", "one_to_many", "1:m",
        "many_to_one", "m:1", "many_to_many", "m:m",
    }:
        raise ValueError("validate must specify a pandas merge cardinality.")
    for side, frame in (("left", left), ("right", right)):
        if not frame.columns.is_unique:
            raise ValueError(f"{side} has duplicate column names.")
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            raise KeyError(f"{side} is missing merge columns: {missing}")

    left_keys, right_keys = left[keys], right[keys]
    null_left = int(left_keys.isna().any(axis=1).sum())
    null_right = int(right_keys.isna().any(axis=1).sum())
    if not allow_null_keys and (null_left or null_right):
        raise ValueError(
            f"[{step_name}] null-key rows: left={null_left}, right={null_right}. "
            "Clean them first or set allow_null_keys=True explicitly."
        )

    # Select an indicator name that cannot collide with a merge key.
    indicator = "_merge"
    while indicator in keys:
        indicator = "_" + indicator
    coverage = left_keys.drop_duplicates().merge(
        right_keys.drop_duplicates(), on=keys, how="outer",
        indicator=indicator, validate="one_to_one",
    )
    left_only = coverage.loc[coverage[indicator].eq("left_only"), keys].copy()
    right_only = coverage.loc[coverage[indicator].eq("right_only"), keys].copy()
    unmatched_left_rows = len(left_keys.merge(left_only, on=keys, how="inner"))
    unmatched_right_rows = len(right_keys.merge(right_only, on=keys, how="inner"))

    print(f"[{step_name}] inputs: left={len(left):,}, right={len(right):,}; "
          f"how={how}, validate={validate}.")
    print(f"[{step_name}] unmatched left: {len(left_only):,} keys / "
          f"{unmatched_left_rows:,} rows; unmatched right: "
          f"{len(right_only):,} keys / {unmatched_right_rows:,} rows.")
    for side, unmatched in (("left", left_only), ("right", right_only)):
        if not unmatched.empty:
            print(f"[{step_name}] {side}-only key examples: "
                  f"{unmatched.head(5).to_dict(orient='records')}")

    merged = left.merge(
        right, on=keys, how=how, validate=validate, suffixes=suffixes,
    )
    retained_left_only = unmatched_left_rows if how in {"left", "outer"} else 0
    added_right_rows = unmatched_right_rows if how in {"right", "outer"} else 0
    excluded_left_rows = unmatched_left_rows - retained_left_only
    matched_output_rows = len(merged) - retained_left_only - added_right_rows
    matched_expansion_rows = matched_output_rows - (len(left) - unmatched_left_rows)
    row_report = track_rows(
        left, merged, step_name,
        reason=(f"{how} join on {keys}: excluded {excluded_left_rows:,} unmatched "
                f"left rows, added {added_right_rows:,} right-only rows, "
                f"added {matched_expansion_rows:,} rows from repeated matches."),
    )
    report = {
        **row_report,
        "how": how,
        "validate": validate,
        "right_rows": len(right),
        "matched_keys": int(coverage[indicator].eq("both").sum()),
        "left_only_keys": left_only.reset_index(drop=True),
        "right_only_keys": right_only.reset_index(drop=True),
        "unmatched_left_rows": unmatched_left_rows,
        "unmatched_right_rows": unmatched_right_rows,
        "excluded_left_rows": excluded_left_rows,
        "added_right_rows": added_right_rows,
        "matched_expansion_rows": matched_expansion_rows,
        "null_left_rows": null_left,
        "null_right_rows": null_right,
        "duplicate_left_rows": int(left_keys.duplicated(keep=False).sum()),
        "duplicate_right_rows": int(right_keys.duplicated(keep=False).sum()),
    }
    print(f"[{step_name}] rows with repeated keys: "
          f"left={report['duplicate_left_rows']:,}, "
          f"right={report['duplicate_right_rows']:,}.")
    return merged, report
