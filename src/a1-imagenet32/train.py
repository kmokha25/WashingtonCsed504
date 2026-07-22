"""
train.py -- train ResNet-18 or a ViT on ImageNet-32 (1.28M images, 1000 classes).

Runs headless for hours, so it checkpoints every epoch and can resume.  The whole dataset lives in
GPU memory (see data.py), so there is no DataLoader, no worker process, and no CPU in the inner loop.

EXAMPLES
    python train.py --model resnet18 --gpu 0            # ~40 min
    python train.py --model vit      --gpu 1            # run concurrently on the other card
    python train.py --model resnet18 --smoke-test       # 30-second sanity check, exits
    python train.py --model vit --resume                # pick up from the last checkpoint

RECIPES (defaults below)
    resnet18: SGD + momentum, LR scaled linearly with batch size from the CIFAR baseline (0.1 @ 128),
              5-epoch warmup, cosine decay, label smoothing 0.1.
    vit:      AdamW, LR 1e-3, weight decay 0.05, 5-epoch warmup, cosine.  Warmup is NOT optional for
              a transformer -- without it the attention softmax saturates on the first noisy batches.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn

import data as D
import engine as E
import models as M

OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "runs",
)

SUBSET_DIR = os.path.join(
    OUT_DIR,
    "subsets",
)


def build_optimizer(model, name, batch_size, args):
    # CNNs get SGD+momentum; transformers get AdamW.  Keying off the model FAMILY, not one exact
    # name -- an earlier version tested `name == 'resnet18'` and so silently handed resnet50 the
    # transformer's AdamW recipe, which would have quietly wrecked that run.
    if name.startswith("resnet"):
        # Linear LR scaling, Goyal et al.: lr = 0.1 * batch/256.
        #
        # GET THIS RIGHT.  The first version of this line said batch/128 -- carrying over CIFAR-10's
        # baseline (0.1 @ batch 128) instead of ImageNet's (0.1 @ batch 256).  That doubled the LR to
        # 0.4, and the result was not a slightly worse model, it was TWO BROKEN RUNS: ResNet-18's
        # validation accuracy peaked at epoch 2 and then fell for the next seven as warmup pushed the
        # LR up, and ResNet-50 -- deeper, less stable -- went to loss=NaN on epoch 1 and never
        # recovered.  A 2x LR error is not a tuning detail; it is the difference between training and
        # not training.
        lr = args.lr if args.lr else 0.1 * batch_size / 256
        opt = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
            fused=True,
        )
    else:
        # AdamW for the transformer.  Note weight_decay=0.05 is NOT comparable to the CNN's 5e-4 --
        # AdamW decouples decay from the gradient, so the number means something different.
        lr = args.lr if args.lr else 1e-3 * batch_size / 512
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.05, fused=True
        )
    return opt, lr


def fraction_token(fraction: float) -> str:
    """Convert a fraction into a readable run-tag component.

    Examples:
        0.05  -> "5"
        0.10  -> "10"
        0.125 -> "12p5"
        1.00  -> "100"
    """
    percent = f"{fraction * 100:.3f}"
    percent = percent.rstrip("0").rstrip(".")
    return percent.replace(".", "p")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(M.BUILDERS), required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help=(
            "proportion of ImageNet-32 training rows to use; "
            "selection is stratified and saved by seed"
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for model initialization and subset selection",
    )

    p.add_argument("--lr", type=float, default=None, help="override the scaled default")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument(
        "--data-parallel",
        action="store_true",
        help="split each batch across BOTH GPUs (measure before trusting it)",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--smoke-test", action="store_true", help="tiny subset, 2 epochs, then exit"
    )
    p.add_argument(
        "--tag", default=None, help="name for this run (default: model name)"
    )
    p.add_argument(
        "--strong-aug",
        dest="strong_aug",
        action="store_true",
        default=None,
        help="mixup + CutMix + random erasing (default: ON for ViTs, OFF for CNNs)",
    )
    p.add_argument("--no-strong-aug", dest="strong_aug", action="store_false")
    p.add_argument(
        "--clip", type=float, default=1.0, help="grad-norm clip (0 to disable)"
    )
    args = p.parse_args()

    if not 0.0 < args.train_fraction <= 1.0:
        p.error("--train-fraction must be in the interval (0, 1].")

    if args.smoke_test and args.train_fraction != 1.0:
        p.error(
            "--smoke-test already uses a fixed 50,000-image subset; "
            "do not combine it with --train-fraction."
        )

    # Smoke tests get their OWN tag.  Reusing the real tag appended 2 bogus 50k-subset epochs to the
    # front of the real run's JSONL, which would have quietly corrupted every plot downstream.
    if args.tag is not None:
        tag = args.tag

    elif args.smoke_test:
        tag = f"smoke-{args.model}-seed{args.seed}"

    elif args.train_fraction < 1.0:
        tag = f"{args.model}-frac{fraction_token(args.train_fraction)}-seed{args.seed}"

    else:
        # Preserve the repository's previous full-dataset default.
        tag = args.model

    # ViTs get mixup/CutMix/erasing by DEFAULT; CNNs do not.  This is not favoritism -- it is what the
    # first run measured: with only crop+flip the ViTs memorized the training set (97% train / 33%
    # val) while the ResNet stayed healthy (+8.7% gap).  The transformer has no locality prior to
    # restrain it, so the regularization has to come from the data instead.
    if args.strong_aug is None:
        args.strong_aug = args.model.startswith("vit")
    device = torch.device(f"cuda:{args.gpu}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The ImageNet-32 training pipeline requires an NVIDIA CUDA GPU. "
            "The subset-generation and reporting tools can run on CPU, but "
            "model training should be run on the team CUDA workstation or "
            "another CUDA-enabled environment."
        )

    torch.cuda.set_device(
        device
    )  # else torch.cuda.synchronize()/memory stats hit device 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, f"{tag}.pt")
    jsonl_path = os.path.join(OUT_DIR, f"{tag}.jsonl")

    epochs = 2 if args.smoke_test else args.epochs

    # Smoke tests retain their established fixed row counts. Real experiments
    # use --train-fraction and always evaluate on the complete validation set.
    train_subset_size = 50_000 if args.smoke_test else None

    train_subset_fraction = (
        args.train_fraction
        if not args.smoke_test and args.train_fraction < 1.0
        else None
    )

    validation_subset_size = 10_000 if args.smoke_test else None

    print(f"[{tag}] device {device} ({torch.cuda.get_device_name(device)})")

    t_load = time.time()

    train_ds = D.GpuImageNet32(
        device,
        split="train",
        subset=train_subset_size,
        subset_fraction=train_subset_fraction,
        seed=args.seed,
        subset_manifest_dir=SUBSET_DIR,
    )

    val_ds = D.GpuImageNet32(
        device,
        split="val",
        subset=validation_subset_size,
        seed=args.seed,
        subset_manifest_dir=SUBSET_DIR,
    )

    print(
        f"[{tag}] dataset resident on GPU: "
        f"{train_ds.gb():.1f} GB train + "
        f"{val_ds.gb():.1f} GB val "
        f"({train_ds.n:,} + {val_ds.n:,} images) "
        f"in {time.time() - t_load:.0f}s"
    )

    print(
        f"[{tag}] training fraction: "
        f"{train_ds.actual_fraction:.3%} "
        f"({train_ds.n:,}/{train_ds.full_n:,})"
    )

    if train_ds.subset_manifest_path is not None:
        print(f"[{tag}] training subset manifest: {train_ds.subset_manifest_path}")

        print(f"[{tag}] subset checksum: {train_ds.subset_metadata['indices_sha256']}")

    print(
        f"[{tag}] dataset resident on GPU: {train_ds.gb():.1f} GB train + {val_ds.gb():.1f} GB val "
        f"({train_ds.n:,} + {val_ds.n:,} images) in {time.time() - t_load:.0f}s"
    )

    model = M.build(args.model, train_ds.n_classes).to(device)
    print(f"[{tag}] {args.model}: {M.n_params(model):,} parameters")
    if args.data_parallel:
        model = nn.DataParallel(model, device_ids=[0, 1])
        print(
            f"[{tag}] DataParallel across GPUs 0 and 1 "
            f"(each card sees a half-batch of {args.batch // 2})"
        )

    optimizer, lr = build_optimizer(model, args.model, args.batch, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda")

    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup
    )
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - args.warmup)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warm, cos], milestones=[args.warmup]
    )

    start_epoch, best, history = 1, 0.0, []
    if not args.resume and os.path.exists(jsonl_path):
        os.remove(
            jsonl_path
        )  # fresh run == fresh history; never append to a previous run's log
    if args.resume and os.path.exists(ckpt_path):
        e, best, history = E.load_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler, device
        )
        start_epoch = e + 1
        print(f"[{tag}] resumed from epoch {e} (best top1 {best:.2%})")

    print(
        f"[{tag}] strong aug (mixup/cutmix/erasing): {args.strong_aug} | grad clip {args.clip}"
    )
    print(
        f"[{tag}] {epochs} epochs, batch {args.batch}, lr {lr:.4f} "
        f"({args.warmup}-epoch warmup -> cosine), label smoothing {args.label_smoothing}"
    )
    print(f"[{tag}] {train_ds.n_batches(args.batch):,} batches/epoch\n", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        tr = E.train_one_epoch(
            model,
            train_ds,
            optimizer,
            criterion,
            scaler,
            device,
            args.batch,
            epoch,
            epochs,
            strong_aug=args.strong_aug,
            clip=(args.clip or None),
        )
        va = E.evaluate(model, val_ds, criterion, device)
        scheduler.step()

        is_best = va["top1"] > best
        best = max(best, va["top1"])
        history.append({"epoch": epoch, "train": tr, "val": va})
        E.log_epoch(
            tag,
            epoch,
            epochs,
            tr,
            va,
            scheduler.get_last_lr()[0],
            time.time() - t_start,
            device,
            is_best,
            jsonl_path,
        )
        E.save_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler, epoch, best, history
        )

    total = time.time() - t_start
    print(f"\n[{tag}] DONE. best val top1 {best:.2%} in {E.fmt_time(total)}")
    result = {
        "tag": tag,
        "model": args.model,
        "params": M.n_params(
            model.module if isinstance(model, nn.DataParallel) else model
        ),
        "epochs": epochs,
        "batch": args.batch,
        "lr": lr,
        "seed": args.seed,
        "requested_train_fraction": args.train_fraction,
        "actual_train_fraction": train_ds.actual_fraction,
        "train_size": train_ds.n,
        "full_train_size": train_ds.full_n,
        "validation_size": val_ds.n,
        "subset_manifest": train_ds.subset_manifest_path,
        "subset_metadata": train_ds.subset_metadata,
        "best_top1": best,
        "seconds": total,
        "history": history,
    }

    result_path = os.path.join(
        OUT_DIR,
        f"{tag}_result.json",
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as result_file:
        json.dump(
            result,
            result_file,
            indent=2,
        )


if __name__ == "__main__":
    main()
