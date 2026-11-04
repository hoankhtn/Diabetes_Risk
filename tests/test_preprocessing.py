"""Regression checks for numeric cleaning and row/merge auditing.

Run from the repository root: python -m unittest discover -s tests -v
"""

from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.preprocessing import (
    clean_numeric_columns,
    detect_outliers,
    filter_outliers,
    replace_placeholders,
)
from src.utils import track_rows, verify_merge


class AuditTests(unittest.TestCase):
    def test_row_loss_growth_and_empty_baseline(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        output = StringIO()
        with redirect_stdout(output):
            loss = track_rows(df, df.iloc[:3], "Drop invalid", reason="Invalid x")
            growth = track_rows(df.iloc[:3], df, "Expand")
            empty = track_rows(df.iloc[:0], df, "Initial load")
            unchanged = track_rows(df, df.copy(), "Replace cells")
        self.assertEqual((loss["change"], loss["change_pct"]), (-1, -25.0))
        self.assertEqual(growth["change"], 1)
        self.assertIsNone(empty["change_pct"])
        self.assertEqual(unchanged["change"], 0)
        self.assertIn("Invalid x", output.getvalue())
        self.assertIn("4 -> 3", output.getvalue())

    def test_inner_merge_audits_keys_discarded_from_both_inputs(self):
        left = pd.DataFrame({"id": [1, 2, 2], "x": [10, 20, 21]})
        right = pd.DataFrame({"id": [1, 3], "y": [30, 40]})
        original_left, original_right = left.copy(), right.copy()
        merged, report = verify_merge(left, right, "id", how="inner")
        self.assertEqual(merged["id"].tolist(), [1])
        self.assertEqual(report["left_only_keys"]["id"].tolist(), [2])
        self.assertEqual(report["right_only_keys"]["id"].tolist(), [3])
        self.assertEqual(report["unmatched_left_rows"], 2)
        self.assertEqual(report["unmatched_right_rows"], 1)
        self.assertEqual(report["duplicate_left_rows"], 2)
        self.assertEqual(report["change"], -2)
        self.assertEqual(report["excluded_left_rows"], 2)
        assert_frame_equal(left, original_left)
        assert_frame_equal(right, original_right)

    def test_composite_reference_join_and_each_join_type(self):
        left = pd.DataFrame({"age_group": ["young", "young", "old"],
                             "gender": ["F", "F", "M"]})
        right = pd.DataFrame({"age_group": ["young", "old"],
                              "gender": ["F", "F"], "median_bmi": [21, 24]})
        for how, count in (("left", 3), ("inner", 2), ("right", 3), ("outer", 4)):
            with self.subTest(how=how):
                merged, report = verify_merge(left, right, ["age_group", "gender"], how=how)
                self.assertEqual(len(merged), count)
                self.assertEqual(report["matched_keys"], 1)
                self.assertEqual(
                    report["change"], report["matched_expansion_rows"]
                    + report["added_right_rows"] - report["excluded_left_rows"],
                )

    def test_cardinality_and_explicit_many_to_many_expansion(self):
        left = pd.DataFrame({"id": [1, 1]})
        right = pd.DataFrame({"id": [1, 1, 1]})
        with self.assertRaises(pd.errors.MergeError):
            verify_merge(left, right, "id")
        merged, report = verify_merge(left, right, "id", validate="many_to_many")
        self.assertEqual(len(merged), 6)
        self.assertEqual(report["change"], 4)
        self.assertEqual(report["matched_expansion_rows"], 4)
        self.assertEqual(report["matched_keys"], 1)

    def test_null_key_policy(self):
        left = pd.DataFrame({"id": [1.0, np.nan]})
        right = pd.DataFrame({"id": [np.nan, 2.0]})
        with self.assertRaisesRegex(ValueError, "null-key rows"):
            verify_merge(left, right, "id")
        merged, report = verify_merge(left, right, "id", allow_null_keys=True)
        self.assertEqual(len(merged), 2)
        self.assertEqual(report["matched_keys"], 1)
        self.assertEqual(report["null_left_rows"], 1)
        self.assertEqual(report["unmatched_left_rows"], 1)

    def test_empty_inputs_and_indicator_key_name(self):
        empty = pd.DataFrame({"_merge": pd.Series(dtype="int64")})
        right = pd.DataFrame({"_merge": [1, 2]})
        merged, report = verify_merge(empty, right, "_merge")
        self.assertTrue(merged.empty)
        self.assertEqual(report["right_only_keys"]["_merge"].tolist(), [1, 2])
        merged, report = verify_merge(right, empty, "_merge")
        self.assertEqual(len(merged), 2)
        self.assertEqual(report["unmatched_left_rows"], 2)
        _, report = verify_merge(empty, empty, "_merge")
        self.assertEqual(report["matched_keys"], 0)

    def test_invalid_merge_arguments(self):
        df = pd.DataFrame({"id": [1]})
        for kwargs in ({"on": []}, {"on": ["id", "id"]},
                       {"on": "id", "how": "cross"},
                       {"on": "id", "validate": None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                verify_merge(df, df, **kwargs)
        with self.assertRaises(KeyError):
            verify_merge(df, pd.DataFrame({"other": [1]}), "id")


class NumericCleaningTests(unittest.TestCase):
    def test_placeholder_replacement_preserves_rows_ids_and_input(self):
        df = pd.DataFrame({"patient_id": [999, 2, 3, 4], "bmi": [-1, 999, 0, 20]})
        original = df.copy(deep=True)
        result = replace_placeholders(df, "bmi")
        self.assertEqual(len(result), 4)
        self.assertEqual(result["bmi"].isna().tolist(), [True, True, False, False])
        self.assertEqual(result["patient_id"].tolist(), [999, 2, 3, 4])
        assert_frame_equal(df, original)
        custom = replace_placeholders(df, "bmi", placeholders=[0])
        self.assertEqual(custom["bmi"].isna().tolist(), [False, False, True, False])

    def test_coercion_and_optional_missing_drop_are_audited(self):
        df = pd.DataFrame({"id": [999, 2, 3, 4, 5, 6, 7],
                           "x": [" 12.5 ", "-1", "999", "bad", np.inf, None, "4"]},
                          index=[10, 11, 12, 13, 14, 15, 16])
        original = df.copy(deep=True)
        output = StringIO()
        with redirect_stdout(output):
            result = clean_numeric_columns(df, "x")
            dropped = clean_numeric_columns(df, "x", drop_missing=True)
        self.assertEqual(result["x"].isna().sum(), 5)
        self.assertEqual(dropped.index.tolist(), [10, 16])
        self.assertEqual(result["id"].iloc[0], 999)
        self.assertIn("7 -> 2", output.getvalue())
        self.assertIn("Drop missing numeric values", output.getvalue())
        assert_frame_equal(df, original)

    def test_iqr_flags_boundaries_missing_and_single_pass_row_union(self):
        df = pd.DataFrame({"x": [0, 1, 2, 3, 6], "y": [1, 2, 3, 4, 100]})
        flags = detect_outliers(df, ["x", "y"])
        self.assertFalse(flags["x"].any())  # 6 equals the upper fence.
        self.assertEqual(flags["y"].tolist(), [False, False, False, False, True])
        df = pd.DataFrame({"x": [1, 2, 3, 4, 100, np.nan],
                           "y": [1, 2, 3, 4, 200, np.nan]}, index=[7, 7, 8, 9, 10, 11])
        result = filter_outliers(df, ["x", "y"])
        self.assertEqual(result.index.tolist(), [7, 7, 8, 9, 11])
        self.assertTrue(result.iloc[-1].isna().all())

    def test_zscore_uses_population_sd_and_strict_threshold(self):
        df = pd.DataFrame({"x": [0.0] * 10 + [100.0, np.nan]})
        flags = detect_outliers(df, "x", method="zscore")
        self.assertEqual(flags["x"].sum(), 1)
        self.assertTrue(flags["x"].iloc[10])
        result = clean_numeric_columns(df, "x", outlier_method="zscore")
        self.assertEqual(len(result), 11)
        boundary = pd.DataFrame({"x": [-1.0, 1.0]})
        self.assertFalse(detect_outliers(boundary, "x", method="zscore", z_threshold=1).any().any())

    def test_degenerate_nullable_and_empty_columns(self):
        df = pd.DataFrame({"constant": pd.Series([2, 2, pd.NA], dtype="Int64"),
                           "missing": [np.nan] * 3, "zero_iqr": [0, 0, 1]})
        for method in ("iqr", "zscore"):
            with self.subTest(method=method):
                flags = detect_outliers(df, ["constant", "missing"], method=method)
                self.assertFalse(flags.any().any())
                self.assertTrue(filter_outliers(df.iloc[:0], "constant", method=method).empty)
        zero_iqr = pd.DataFrame({"x": [0] * 9 + [1]})
        self.assertFalse(detect_outliers(zero_iqr, "x").any().any())
        result = replace_placeholders(pd.DataFrame({"x": pd.Series([-1, 2], dtype="Int64")}), "x")
        self.assertTrue(pd.isna(result["x"].iloc[0]))

    def test_invalid_arguments_and_infinity_require_explicit_cleaning(self):
        df = pd.DataFrame({"x": [1, 2]})
        for kwargs in ({"method": "invalid"}, {"iqr_multiplier": 0},
                       {"method": "zscore", "z_threshold": np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                detect_outliers(df, "x", **kwargs)
        with self.assertRaises(KeyError):
            clean_numeric_columns(df, "absent")
        with self.assertRaises(ValueError):
            clean_numeric_columns(df, [])
        with self.assertRaises(TypeError):
            detect_outliers(pd.DataFrame({"x": ["1", "2"]}), "x")
        with self.assertRaises(ValueError):
            detect_outliers(pd.DataFrame({"x": [1, np.inf]}), "x")


if __name__ == "__main__":
    unittest.main()
