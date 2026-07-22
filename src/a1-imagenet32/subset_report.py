"""Generate validation reports for ImageNet-32 stratified subsets.

This script does not train a model or load image tensors onto a GPU. It only
loads the label array, creates/reuses subset manifests, and measures how well
each subset preserves the original class distribution.

Example:
    python subset_report.py --fractions 0.05 0.10 0.25 0.50 1.0 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from subsets import get_or_create_stratified_indices


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS_PATH = SCRIPT_DIR / "data" / "train_y.npy"
DEFAULT_MANIFEST_DIR = SCRIPT_DIR / "runs" / "subsets"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "runs" / "subset_reports"


def build_demo_labels() -> np.ndarray:
    """Return a small imbalanced label array for CPU-only demonstration.

    This dataset is synthetic and should only be used to validate the reporting
    pipeline. Results from demo mode are not ImageNet experiment results.
    """
    class_counts = np.array(
        [100, 120, 150, 180, 220, 260, 310, 370, 440, 520],
        dtype=np.int64,
    )

    return np.repeat(
        np.arange(len(class_counts), dtype=np.int64),
        class_counts,
    )


def aligned_class_counts(
    labels: np.ndarray,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return class IDs and aligned full/subset counts."""
    classes, full_counts = np.unique(
        labels,
        return_counts=True,
    )

    selected_labels = labels[selected_indices]

    selected_classes, selected_counts_raw = np.unique(
        selected_labels,
        return_counts=True,
    )

    selected_count_map = dict(
        zip(
            selected_classes.tolist(),
            selected_counts_raw.tolist(),
            strict=True,
        )
    )

    selected_counts = np.array(
        [
            selected_count_map.get(
                int(class_id),
                0,
            )
            for class_id in classes
        ],
        dtype=np.int64,
    )

    return classes, full_counts, selected_counts


def summarize_subset(
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    requested_fraction: float,
    seed: int,
    checksum: str,
    manifest_path: str | None,
) -> dict[str, Any]:
    """Calculate distribution and reproducibility statistics."""
    classes, full_counts, subset_counts = aligned_class_counts(
        labels,
        indices,
    )

    full_distribution = full_counts / full_counts.sum()
    subset_distribution = subset_counts / subset_counts.sum()

    absolute_distribution_error = np.abs(
        subset_distribution - full_distribution
    )

    # Total variation distance is zero for identical distributions and
    # approaches one as distributions diverge.
    total_variation_distance = float(
        0.5 * absolute_distribution_error.sum()
    )

    return {
        "requested_fraction": requested_fraction,
        "actual_fraction": float(len(indices) / len(labels)),
        "seed": seed,
        "selected_images": int(len(indices)),
        "full_dataset_images": int(len(labels)),
        "classes_in_full_dataset": int(len(classes)),
        "classes_in_subset": int(np.count_nonzero(subset_counts)),
        "missing_classes": int(np.count_nonzero(subset_counts == 0)),
        "minimum_examples_per_class": int(subset_counts.min()),
        "maximum_examples_per_class": int(subset_counts.max()),
        "mean_examples_per_class": float(subset_counts.mean()),
        "total_variation_distance": total_variation_distance,
        "maximum_class_share_error": float(
            absolute_distribution_error.max()
        ),
        "indices_sha256": checksum,
        "manifest_path": manifest_path,
    }


def calculate_overlap(
    first_indices: np.ndarray,
    second_indices: np.ndarray,
) -> tuple[int, float]:
    """Measure selected-row overlap between two different seeds."""
    overlap_count = int(
        np.intersect1d(
            first_indices,
            second_indices,
            assume_unique=True,
        ).size
    )

    overlap_fraction = overlap_count / len(first_indices)

    return overlap_count, overlap_fraction


def print_report(rows: list[dict[str, Any]]) -> None:
    """Print a compact table suitable for terminal output."""
    print()
    print("ImageNet-32 stratified subset validation")
    print("=" * 112)

    header = (
        f"{'Fraction':>9} "
        f"{'Images':>10} "
        f"{'Classes':>9} "
        f"{'Min/Class':>10} "
        f"{'Max/Class':>10} "
        f"{'Mean/Class':>11} "
        f"{'TVD':>10} "
        f"{'Max Error':>11} "
        f"{'Seed Overlap':>13}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        class_display = (
            f"{row['classes_in_subset']}/"
            f"{row['classes_in_full_dataset']}"
        )

        print(
            f"{row['actual_fraction']:>8.1%} "
            f"{row['selected_images']:>10,} "
            f"{class_display:>9} "
            f"{row['minimum_examples_per_class']:>10,} "
            f"{row['maximum_examples_per_class']:>10,} "
            f"{row['mean_examples_per_class']:>11.2f} "
            f"{row['total_variation_distance']:>10.6f} "
            f"{row['maximum_class_share_error']:>11.8f} "
            f"{row['alternate_seed_overlap_fraction']:>12.2%}"
        )

    print("=" * 112)
    print(
        "TVD = total variation distance between the complete dataset and "
        "subset class distributions. Lower is better."
    )
    print(
        "Seed overlap = proportion of selected images also selected using "
        "the alternate seed."
    )
    print()


def save_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save one summary row per requested fraction."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    labels_path: str,
    demo_mode: bool,
) -> None:
    """Save the report and top-level context as JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "demo_mode": demo_mode,
        "labels_path": labels_path,
        "rows": rows,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            payload,
            json_file,
            indent=2,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate deterministic stratified subsets without training."
        )
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="path to the ImageNet-32 training label array",
    )

    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25, 0.50, 1.0],
        help="training fractions to validate",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="primary subset-selection seed",
    )

    parser.add_argument(
        "--compare-seed",
        type=int,
        default=43,
        help="alternate seed used to measure subset overlap",
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="directory used to save exact subset-index manifests",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="directory used to save CSV and JSON reports",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="use a small synthetic label array instead of ImageNet labels",
    )

    return parser.parse_args()


def main() -> None:
    """Create manifests, validate them, print results, and save reports."""
    args = parse_arguments()

    for fraction in args.fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"Every fraction must be in (0, 1], got {fraction}."
            )

    if args.seed == args.compare_seed:
        raise ValueError(
            "--seed and --compare-seed must be different."
        )

    if args.demo:
        labels = build_demo_labels()
        labels_source = "synthetic demonstration labels"
        report_prefix = "demo"
    else:
        if not args.labels.exists():
            raise FileNotFoundError(
                f"Could not find ImageNet-32 labels at:\n"
                f"  {args.labels}\n\n"
                "Run prepare_data.py first, obtain train_y.npy from a "
                "teammate, or use --demo to validate the report pipeline."
            )

        labels = np.load(
            args.labels,
            mmap_mode="r",
        )

        labels_source = str(args.labels)
        report_prefix = "imagenet32"

    rows: list[dict[str, Any]] = []

    print(f"Labels source: {labels_source}")
    print(f"Rows: {len(labels):,}")
    print(f"Primary seed: {args.seed}")
    print(f"Comparison seed: {args.compare_seed}")

    for fraction in sorted(args.fractions):
        primary_indices, metadata, manifest_path = (
            get_or_create_stratified_indices(
                labels,
                fraction=fraction,
                seed=args.seed,
                split="train",
                manifest_dir=args.manifest_dir,
                ensure_all_classes=True,
            )
        )

        alternate_indices, _, _ = (
            get_or_create_stratified_indices(
                labels,
                fraction=fraction,
                seed=args.compare_seed,
                split="train",
                manifest_dir=args.manifest_dir,
                ensure_all_classes=True,
            )
        )

        overlap_count, overlap_fraction = calculate_overlap(
            primary_indices,
            alternate_indices,
        )

        row = summarize_subset(
            labels,
            primary_indices,
            requested_fraction=fraction,
            seed=args.seed,
            checksum=metadata["indices_sha256"],
            manifest_path=(
                str(manifest_path)
                if manifest_path is not None
                else None
            ),
        )

        row["alternate_seed"] = args.compare_seed
        row["alternate_seed_overlap_count"] = overlap_count
        row["alternate_seed_overlap_fraction"] = overlap_fraction

        rows.append(row)

    print_report(rows)

    csv_path = (
        args.output_dir
        / f"{report_prefix}_subset_report_seed{args.seed}.csv"
    )

    json_path = (
        args.output_dir
        / f"{report_prefix}_subset_report_seed{args.seed}.json"
    )

    save_csv(
        rows,
        csv_path,
    )

    save_json(
        rows,
        json_path,
        labels_path=labels_source,
        demo_mode=args.demo,
    )

    print(f"CSV report:  {csv_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()