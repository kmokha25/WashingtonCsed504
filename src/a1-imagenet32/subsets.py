"""Deterministic stratified subset utilities for ImageNet-32 experiments.

The scaling study must compare ResNet and ViT on the exact same training
examples. This module creates a class-stratified subset once, saves its indices,
and reuses that manifest for every model trained with the same size and seed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_VERSION = 1


def _sha256_array(array: np.ndarray) -> str:
    """Return a stable SHA-256 digest for a contiguous NumPy array."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def resolve_target_size(
    dataset_size: int,
    *,
    fraction: float | None = None,
    size: int | None = None,
) -> int:
    """Convert a requested fraction or count into an exact subset size.

    Exactly one of ``fraction`` and ``size`` must be supplied.

    Fractions are rounded to the nearest image because a dataset cannot contain
    a fractional row.
    """
    if (fraction is None) == (size is None):
        raise ValueError("Provide exactly one of fraction or size.")

    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"fraction must be in the interval (0, 1], got {fraction}."
            )

        target_size = int(np.floor(dataset_size * fraction + 0.5))

    else:
        assert size is not None
        target_size = int(size)

    if not 1 <= target_size <= dataset_size:
        raise ValueError(
            f"subset size must be between 1 and {dataset_size:,}, got {target_size:,}."
        )

    return target_size


def allocate_stratified_counts(
    class_counts: np.ndarray,
    target_size: int,
    *,
    ensure_all_classes: bool = True,
) -> np.ndarray:
    """Allocate an exact sample count to every class.

    The selected subset retains the original class proportions rather than
    forcing every class to contain an equal number of images.

    When the target contains at least as many images as there are classes, one
    image is reserved for every class. The remainder is distributed
    proportionally using the largest-remainder apportionment method.
    """
    counts = np.asarray(class_counts, dtype=np.int64)

    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("class_counts must be a non-empty 1-D array.")

    if np.any(counts <= 0):
        raise ValueError("class_counts must contain only positive values.")

    total = int(counts.sum())

    if not 1 <= target_size <= total:
        raise ValueError(
            f"target_size must be between 1 and {total:,}, got {target_size:,}."
        )

    number_of_classes = counts.size

    # Reserve one example for every class whenever the requested subset is
    # large enough to contain all classes.
    minimum = np.zeros_like(counts)

    if ensure_all_classes and target_size >= number_of_classes:
        minimum[:] = 1

    remaining_target = target_size - int(minimum.sum())
    remaining_capacity = counts - minimum

    if remaining_target == 0:
        return minimum

    capacity_total = int(remaining_capacity.sum())

    # Calculate the ideal fractional allocation for each class.
    quotas = remaining_capacity * (remaining_target / capacity_total)

    # Start with the integer portion of every quota.
    allocation = minimum + np.floor(quotas).astype(np.int64)

    # Flooring normally leaves several rows unallocated. Give those rows to
    # classes with the largest fractional remainders.
    leftover = target_size - int(allocation.sum())

    if leftover:
        fractional_remainders = quotas - np.floor(quotas)

        # Stable sorting means tied classes are resolved deterministically.
        candidates = np.argsort(
            -fractional_remainders,
            kind="stable",
        )

        # A class cannot receive more rows than actually exist.
        candidates = candidates[allocation[candidates] < counts[candidates]]

        allocation[candidates[:leftover]] += 1

    if int(allocation.sum()) != target_size:
        raise RuntimeError("Stratified allocation did not reach the requested size.")

    if np.any(allocation > counts):
        raise RuntimeError("Stratified allocation exceeded a class population.")

    return allocation


def build_stratified_indices(
    labels: np.ndarray,
    *,
    target_size: int,
    seed: int,
    ensure_all_classes: bool = True,
) -> np.ndarray:
    """Select deterministic, class-stratified row indices.

    The returned indices are sorted for efficient indexing into the memory-
    mapped image array.

    Sorting these indices does not remove training randomness. The existing
    ``GpuImageNet32.epoch`` method still randomly permutes the selected rows at
    the beginning of every training epoch.
    """
    y = np.asarray(labels)

    if y.ndim != 1:
        raise ValueError(
            f"labels must be a one-dimensional array, got shape {y.shape}."
        )

    # Group row indices by class once.
    #
    # This avoids repeatedly scanning all 1.28 million labels for each of the
    # 1,000 ImageNet classes.
    rows_by_class = np.argsort(y, kind="stable")
    sorted_labels = y[rows_by_class]

    _, starts, counts = np.unique(
        sorted_labels,
        return_index=True,
        return_counts=True,
    )

    allocation = allocate_stratified_counts(
        counts,
        target_size,
        ensure_all_classes=ensure_all_classes,
    )

    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []

    for start, count, take in zip(
        starts,
        counts,
        allocation,
        strict=True,
    ):
        if take == 0:
            continue

        class_rows = rows_by_class[start : start + count]

        if take == count:
            # Avoid unnecessary random sampling when retaining every row from
            # the class.
            chosen = class_rows
        else:
            chosen = rng.choice(
                class_rows,
                size=int(take),
                replace=False,
            )

        selected_parts.append(np.asarray(chosen, dtype=np.int64))

    indices = np.sort(np.concatenate(selected_parts))

    if len(indices) != target_size:
        raise RuntimeError(
            f"Expected {target_size:,} selected rows, produced {len(indices):,}."
        )

    if len(np.unique(indices)) != target_size:
        raise RuntimeError("The generated subset contains duplicate row indices.")

    return indices


def _manifest_metadata(
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    split: str,
    seed: int,
    requested_fraction: float | None,
) -> dict[str, Any]:
    """Build metadata that describes and verifies a subset manifest."""
    selected_labels = labels[indices]

    _, selected_counts = np.unique(
        selected_labels,
        return_counts=True,
    )

    return {
        "version": MANIFEST_VERSION,
        "strategy": "proportional_stratified_without_replacement",
        "split": split,
        "seed": int(seed),
        "requested_fraction": requested_fraction,
        "dataset_size": int(len(labels)),
        "target_size": int(len(indices)),
        "actual_fraction": float(len(indices) / len(labels)),
        "num_classes": int(np.unique(labels).size),
        "min_selected_per_class": int(selected_counts.min()),
        "max_selected_per_class": int(selected_counts.max()),
        "labels_sha256": _sha256_array(np.asarray(labels)),
        "indices_sha256": _sha256_array(indices),
    }


def get_or_create_stratified_indices(
    labels: np.ndarray,
    *,
    fraction: float | None = None,
    size: int | None = None,
    seed: int = 42,
    split: str = "train",
    manifest_dir: str | os.PathLike[str] | None = None,
    ensure_all_classes: bool = True,
) -> tuple[np.ndarray, dict[str, Any], Path | None]:
    """Load a matching subset manifest or create one deterministically.

    Returns:
        A tuple containing:

        1. Selected row indices.
        2. Subset metadata.
        3. The manifest path, or ``None`` when no directory was provided.

    The scaling experiment should always provide ``manifest_dir``. Saving the
    manifest guarantees that both architectures consume the exact same rows,
    rather than relying only on matching seeds.
    """
    y = np.asarray(labels)

    target_size = resolve_target_size(
        len(y),
        fraction=fraction,
        size=size,
    )

    manifest_path: Path | None = None

    if manifest_dir is not None:
        manifest_root = Path(manifest_dir)
        manifest_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path = manifest_root / (
            f"{split}_stratified_n{target_size:07d}_seed{seed:04d}.npz"
        )

    labels_hash = _sha256_array(y)

    # Reuse a previously created subset when it matches this request.
    if manifest_path is not None and manifest_path.exists():
        with np.load(
            manifest_path,
            allow_pickle=False,
        ) as saved:
            indices = saved["indices"].astype(
                np.int64,
                copy=False,
            )

            metadata = json.loads(str(saved["metadata"].item()))

        expected = {
            "version": MANIFEST_VERSION,
            "split": split,
            "seed": int(seed),
            "dataset_size": int(len(y)),
            "target_size": int(target_size),
            "labels_sha256": labels_hash,
        }

        mismatches = {
            key: (metadata.get(key), expected_value)
            for key, expected_value in expected.items()
            if metadata.get(key) != expected_value
        }

        if mismatches:
            raise ValueError(
                f"Subset manifest {manifest_path} does not match "
                f"the requested experiment: {mismatches}"
            )

        if _sha256_array(indices) != metadata.get("indices_sha256"):
            raise ValueError(f"Subset manifest {manifest_path} failed its checksum.")

        return indices, metadata, manifest_path

    # No matching file exists, so generate the subset.
    indices = build_stratified_indices(
        y,
        target_size=target_size,
        seed=seed,
        ensure_all_classes=ensure_all_classes,
    )

    metadata = _manifest_metadata(
        y,
        indices,
        split=split,
        seed=seed,
        requested_fraction=fraction,
    )

    if manifest_path is not None:
        # Write to a temporary file first, then atomically rename it. This
        # prevents an interrupted run from leaving a corrupt manifest.
        temporary_path = manifest_path.with_name(manifest_path.name + ".tmp.npz")

        np.savez_compressed(
            temporary_path,
            indices=indices,
            metadata=np.array(
                json.dumps(
                    metadata,
                    sort_keys=True,
                )
            ),
        )

        os.replace(
            temporary_path,
            manifest_path,
        )

    return indices, metadata, manifest_path
