"""Unit tests for deterministic stratified subset creation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from subsets import (
    allocate_stratified_counts,
    build_stratified_indices,
    get_or_create_stratified_indices,
    resolve_target_size,
)


class StratifiedSubsetTests(unittest.TestCase):
    """Tests that do not require ImageNet or a GPU."""

    def setUp(self) -> None:
        # Synthetic dataset:
        #
        # class 0: 10 rows
        # class 1: 20 rows
        # class 2: 30 rows
        # class 3: 40 rows
        self.labels = np.repeat(
            np.arange(4),
            [10, 20, 30, 40],
        )

    def test_resolve_fraction(self) -> None:
        self.assertEqual(
            resolve_target_size(
                100,
                fraction=0.25,
            ),
            25,
        )

    def test_exact_proportional_allocation(self) -> None:
        allocation = allocate_stratified_counts(
            np.array([10, 20, 30, 40]),
            target_size=50,
        )

        np.testing.assert_array_equal(
            allocation,
            [5, 10, 15, 20],
        )

    def test_same_seed_produces_same_indices(self) -> None:
        first = build_stratified_indices(
            self.labels,
            target_size=50,
            seed=42,
        )

        second = build_stratified_indices(
            self.labels,
            target_size=50,
            seed=42,
        )

        np.testing.assert_array_equal(
            first,
            second,
        )

    def test_different_seed_changes_rows_not_counts(self) -> None:
        first = build_stratified_indices(
            self.labels,
            target_size=50,
            seed=1,
        )

        second = build_stratified_indices(
            self.labels,
            target_size=50,
            seed=2,
        )

        self.assertFalse(np.array_equal(first, second))

        first_counts = np.bincount(
            self.labels[first],
            minlength=4,
        )

        second_counts = np.bincount(
            self.labels[second],
            minlength=4,
        )

        np.testing.assert_array_equal(
            first_counts,
            second_counts,
        )

    def test_every_class_is_present_when_possible(self) -> None:
        indices = build_stratified_indices(
            self.labels,
            target_size=4,
            seed=42,
        )

        self.assertEqual(
            set(self.labels[indices]),
            {0, 1, 2, 3},
        )

    def test_manifest_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, first_metadata, manifest_path = get_or_create_stratified_indices(
                self.labels,
                fraction=0.5,
                seed=42,
                manifest_dir=directory,
            )

            self.assertIsNotNone(manifest_path)
            assert manifest_path is not None

            self.assertTrue(Path(manifest_path).exists())

            second, second_metadata, second_path = get_or_create_stratified_indices(
                self.labels,
                fraction=0.5,
                seed=42,
                manifest_dir=directory,
            )

            np.testing.assert_array_equal(
                first,
                second,
            )

            self.assertEqual(
                first_metadata,
                second_metadata,
            )

            self.assertEqual(
                manifest_path,
                second_path,
            )

    def test_invalid_request_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_target_size(
                100,
                fraction=0.0,
            )

        with self.assertRaises(ValueError):
            resolve_target_size(
                100,
                fraction=0.5,
                size=50,
            )

    def test_full_subset_returns_every_row(self) -> None:
        """A 100% subset should contain every source row exactly once."""
        indices = build_stratified_indices(
            self.labels,
            target_size=len(self.labels),
            seed=42,
        )

        np.testing.assert_array_equal(
            indices,
            np.arange(len(self.labels)),
        )

    def test_subset_preserves_expected_class_proportions(self) -> None:
        """A 50% subset should preserve this known synthetic distribution."""
        indices = build_stratified_indices(
            self.labels,
            target_size=50,
            seed=42,
        )

        selected_counts = np.bincount(
            self.labels[indices],
            minlength=4,
        )

        np.testing.assert_array_equal(
            selected_counts,
            [5, 10, 15, 20],
        )

if __name__ == "__main__":
    unittest.main()
