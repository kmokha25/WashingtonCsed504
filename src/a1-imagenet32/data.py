"""
data.py -- GPU-resident ImageNet-32 pipeline.

THE IDEA
--------
On CIFAR-10 we measured that the CPU augmentation pipeline was starving the GPU by 3x: one core
augments ~4,000 img/s while the card trains at ~13,000 img/s, so the GPU spent most of its life
waiting for Python.  The standard fix is more DataLoader workers.  Here we can do better and delete
the problem outright.

ImageNet-32's whole training set, as raw uint8, is:

    1,281,167 x 32 x 32 x 3 = 3.9 GB

The RTX PRO 6000 has 96 GB.  So we upload the ENTIRE DATASET to the GPU once, at startup, and
generate batches there.  Augmentation (random crop + horizontal flip) becomes a handful of tensor
ops on data that is already in VRAM.  The result:

  * no DataLoader, no worker processes, no 'spawn' problems, no persistent_workers
  * no PIL, no JPEG decode, no Python in the inner loop
  * no host->device copy per batch -- the data never leaves the GPU
  * the CPU is left completely free

This is the same trick as the GPU-resident hyperparameter sweep in CSED 503 A1, applied to images.
It only works because the images are tiny; you could not do this with 224x224 ImageNet (which would
be ~190 GB).  32x32 is what makes it possible.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import subsets as S

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_stats() -> dict:
    p = os.path.join(DATA_DIR, "stats.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found -- run prepare_data.py first.")
    return json.load(open(p))


class GpuImageNet32:
    """The whole dataset, resident in GPU memory, serving augmented batches.

    Replaces torchvision transforms + Dataset + DataLoader with ~30 lines of tensor code.
    """

    def __init__(
        self,
        device: torch.device,
        split: str = "train",
        subset: int | None = None,
        seed: int = 0,
        subset_fraction: float | None = None,
        subset_manifest_dir: str | None = None,
    ):
        """Load an ImageNet-32 split and optionally select a stratified subset.

        Args:
            device:
                CUDA device that will store the selected images.

            split:
                Either ``"train"`` or ``"val"``.

            subset:
                Optional exact number of rows to retain. This is primarily used
                by the existing smoke-test mode.

            seed:
                Seed used to select subset rows.

            subset_fraction:
                Optional fraction of the split to retain. For example, ``0.05``
                retains approximately 5% of the training set.

            subset_manifest_dir:
                Directory where exact selected row indices will be saved and
                reused.
        """
        stats = load_stats()

        x = np.load(
            os.path.join(DATA_DIR, f"{split}_x.npy"),
            mmap_mode="r",
        )

        y = np.load(
            os.path.join(DATA_DIR, f"{split}_y.npy"),
            mmap_mode="r",
        )

        self.split = split
        self.full_n = len(y)
        self.subset_metadata = None
        self.subset_manifest_path = None

        if subset is not None and subset_fraction is not None:
            raise ValueError("Provide either subset or subset_fraction, not both.")

        if subset is not None and not 1 <= subset <= self.full_n:
            raise ValueError(
                f"subset must be between 1 and {self.full_n:,}, got {subset:,}."
            )

        if subset_fraction is not None and not 0.0 < subset_fraction <= 1.0:
            raise ValueError("subset_fraction must be in the interval (0, 1].")

        use_size_subset = subset is not None and subset < self.full_n

        use_fraction_subset = subset_fraction is not None and subset_fraction < 1.0

        if use_size_subset or use_fraction_subset:
            indices, metadata, manifest_path = S.get_or_create_stratified_indices(
                y,
                size=subset if use_size_subset else None,
                fraction=(subset_fraction if use_fraction_subset else None),
                seed=seed,
                split=split,
                manifest_dir=subset_manifest_dir,
                ensure_all_classes=True,
            )

            x = x[indices]
            y = y[indices]

            self.subset_metadata = metadata
            self.subset_manifest_path = (
                str(manifest_path) if manifest_path is not None else None
            )

        # Keep images as uint8 while resident on the GPU.
        #
        # Image values are converted to normalized floating point tensors only
        # when each batch is requested.
        self.x = torch.from_numpy(np.ascontiguousarray(x)).to(
            device,
            non_blocking=True,
        )

        self.y = (
            torch.from_numpy(np.ascontiguousarray(y))
            .to(
                device,
                non_blocking=True,
            )
            .long()
        )

        self.device = device
        self.n = len(self.y)
        self.actual_fraction = self.n / self.full_n

        # Shape normalization constants as (1, 3, 1, 1), allowing them to
        # broadcast across every image in a batch.
        self.mean = torch.tensor(
            stats["mean"],
            device=device,
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            stats["std"],
            device=device,
        ).view(1, 3, 1, 1)

        self.n_classes = stats["n_classes"]

    def gb(self) -> float:
        return (self.x.numel() + self.y.numel() * 8) / 1e9

    def _to_float(self, xb: torch.Tensor) -> torch.Tensor:
        """(B,32,32,3) uint8 -> (B,3,32,32) normalized float.  This is ToTensor + Normalize."""
        xb = (
            xb.permute(0, 3, 1, 2).float().div_(255.0)
        )  # NHWC uint8 -> NCHW float in [0,1]
        return xb.sub_(self.mean).div_(self.std)  # (x - mean) / std

    def _augment(self, xb: torch.Tensor) -> torch.Tensor:
        """RandomCrop(32, padding=4) + RandomHorizontalFlip, batched, on the GPU.

        Exactly the two transforms from the CIFAR notebook -- just expressed as tensor ops on a whole
        batch at once instead of PIL calls on one image at a time.
        """
        b, d = xb.shape[0], xb.device

        # --- RandomCrop(32, padding=4): pad to 40x40, then cut a random 32x32 window per image.
        xp = F.pad(xb, (4, 4, 4, 4))  # (B,3,40,40), zero padding
        i = torch.randint(0, 9, (b,), device=d)  # 9 = 40 - 32 + 1 valid offsets
        j = torch.randint(0, 9, (b,), device=d)
        ar = torch.arange(32, device=d)
        rows = i.view(b, 1, 1) + ar.view(1, 32, 1)  # (B,32,1)
        cols = j.view(b, 1, 1) + ar.view(1, 1, 32)  # (B,1,32)
        bidx = torch.arange(b, device=d).view(b, 1, 1)
        # Advanced indexing with a slice between the index tensors puts the indexed dims first,
        # so this comes back as (B,32,32,C) -- hence the permute.
        xb = xp[bidx, :, rows, cols].permute(0, 3, 1, 2).contiguous()

        # --- RandomHorizontalFlip(): mirror half the batch.
        flip = torch.rand(b, device=d) < 0.5
        xb[flip] = xb[flip].flip(-1)
        return xb

    def epoch(
        self, batch_size: int, train: bool, generator: torch.Generator | None = None
    ):
        """Yield (images, labels) for one pass. Shuffled + augmented when train=True."""
        if train:
            order = torch.randperm(self.n, device=self.device, generator=generator)
        else:
            order = torch.arange(self.n, device=self.device)

        for s in range(0, self.n, batch_size):
            idx = order[s : s + batch_size]
            xb = self._to_float(self.x[idx])
            if train:
                xb = self._augment(xb)
            yield xb, self.y[idx]

    def n_batches(self, batch_size: int) -> int:
        return (self.n + batch_size - 1) // batch_size


# ---------------------------------------------------------------------------------------------------
# Strong augmentation for the transformers -- all of it on the GPU, all of it pure tensor math.
#
# WHY ONLY THE ViTs NEED THIS.  Our first ImageNet-32 run gave the ViTs only crop+flip, the same as
# the CNN, and they MEMORIZED the dataset: 97.4% train accuracy against 32.6% validation, a 65-point
# gap, on 1.28M images.  The ResNet, with the same augmentation, sat at a healthy +8.7% gap.
#
# That contrast IS the inductive bias, made visible.  A CNN cannot easily memorize -- weight sharing
# and locality physically restrict what it can represent, so the architecture regularizes itself for
# free.  A ViT has no such constraint: every token can attend to every other, so it has more than
# enough capacity to just learn the training set.  Augmentation is the SUBSTITUTE for the prior the
# transformer does not have.  This is why every real ViT recipe (DeiT and successors) ships with
# mixup + CutMix + erasing, and why a CNN recipe can get away without them.
# ---------------------------------------------------------------------------------------------------


def random_erasing_(
    x: torch.Tensor, p: float = 0.25, scale=(0.02, 0.2)
) -> torch.Tensor:
    """Cut a random rectangle out of a random subset of the batch. GPU-side RandomErasing.

    FULLY VECTORIZED -- and it has to be.  The first version looped in Python over the ~25% of the
    batch that gets erased and did one slice-assign per image; that dropped ViT throughput from
    14.3k to 4.2k img/s (3.4x), because a few hundred tiny GPU ops per batch is all launch overhead.
    Building the erase mask with broadcast comparisons instead makes it ~5 kernels for the whole
    batch, regardless of batch size.  Same lesson as the CIFAR launch-bound analysis: on a GPU, a
    Python loop over samples is the enemy.
    """
    b, _, h, w = x.shape
    d = x.device
    area = h * w * (torch.rand(b, device=d) * (scale[1] - scale[0]) + scale[0])
    ratio = torch.empty(b, device=d).uniform_(0.3, 3.3)
    eh = (area * ratio).sqrt().clamp(1, h - 1).long()  # (B,)
    ew = (area / ratio).sqrt().clamp(1, w - 1).long()
    top = (torch.rand(b, device=d) * (h - eh).float()).long()
    left = (torch.rand(b, device=d) * (w - ew).float()).long()

    rows = torch.arange(h, device=d).view(1, h, 1)  # broadcast against (B,1,1)
    cols = torch.arange(w, device=d).view(1, 1, w)
    inside = (
        (rows >= top.view(b, 1, 1))
        & (rows < (top + eh).view(b, 1, 1))
        & (cols >= left.view(b, 1, 1))
        & (cols < (left + ew).view(b, 1, 1))
    )
    hit = (torch.rand(b, device=d) < p).view(b, 1, 1)
    mask = (inside & hit).unsqueeze(1)  # (B,1,H,W) -> broadcasts over C
    return x.masked_fill_(mask, 0.0)  # 0 == the channel mean, post-normalize


def mixup_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    prob: float = 0.5,
):
    """Blend each image with another from the same batch. Returns (x, y_a, y_b, lam).

    mixup:  a weighted pixel average of two images; the target becomes the same weighted mix.
    CutMix: paste a patch of image B into image A; the target mixes by the AREA of the patch.

    The loss then becomes  lam * CE(out, y_a) + (1 - lam) * CE(out, y_b)  -- i.e. the model is never
    shown a confident one-hot target, so it cannot simply memorize image->label pairs.
    """
    b = x.size(0)
    perm = torch.randperm(b, device=x.device)

    if torch.rand(()) < prob:  # ---- mixup
        lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
        x = lam * x + (1 - lam) * x[perm]
    else:  # ---- CutMix
        lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample())
        h, w = x.shape[2:]
        rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        if rh > 0 and rw > 0:
            cy = int(torch.randint(0, h - rh + 1, ()))
            cx = int(torch.randint(0, w - rw + 1, ()))
            x[:, :, cy : cy + rh, cx : cx + rw] = x[perm][
                :, :, cy : cy + rh, cx : cx + rw
            ]
            lam = 1 - (rh * rw) / (h * w)  # true mixed area, not the sampled lam
        else:
            lam = 1.0
    return x, y, y[perm], lam
