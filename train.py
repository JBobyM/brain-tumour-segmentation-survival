"""Phase 2 — Train a 3D U-Net for BraTS tumour segmentation with Dice Loss.

Multi-label setup: 3 overlapping regions (TC, WT, ET) -> sigmoid + DiceLoss.
Validation uses sliding-window inference over full volumes and reports the Dice
score per region. Uses AMP and DataParallel across all visible GPUs.

Examples:
    .venv/bin/python train.py --smoke            # 2 iters, confirm it runs
    .venv/bin/python train.py --epochs 100       # full run
"""

import argparse
import time
from pathlib import Path

import torch
import wandb

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet
from monai.transforms import Activations, AsDiscrete, Compose

from data_pipeline import ROI_SIZE, get_dataloaders

CKPT_DIR = Path("./checkpoints")
REGIONS = ["TC", "WT", "ET"]


def build_model(arch, device):
    if arch == "unet":
        model = UNet(
            spatial_dims=3, in_channels=4, out_channels=3,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
            num_res_units=2, norm="instance",
        )
    elif arch == "segresnet":
        model = SegResNet(
            spatial_dims=3, in_channels=4, out_channels=3,
            init_filters=32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        )
    else:
        raise ValueError(f"unknown arch: {arch}")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"[train] DataParallel across {torch.cuda.device_count()} GPUs")
    return model


def ckpt_path(arch):
    # keep the UNet baseline at the original path; segresnet gets its own file
    return CKPT_DIR / ("best_model.pth" if arch == "unet" else f"best_model_{arch}.pth")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-rate", type=float, default=0.0)
    p.add_argument("--sw-batch", type=int, default=2, help="sliding-window batch size at val")
    p.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    p.add_argument("--arch", type=str, default="unet", choices=["unet", "segresnet"])
    p.add_argument("--wandb-project", type=str, default="brain-tumour-segmentation")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="'disabled' turns off W&B; 'offline' logs locally to sync later",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode if not args.smoke else "disabled",
        config={
            "arch": args.arch,
            "in_channels": 4,
            "out_channels": 3,
            "regions": REGIONS,
            "loss": "DiceLoss(sigmoid)",
            "optimizer": "Adam",
            "lr": args.lr,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "roi_size": ROI_SIZE,
            "cache_rate": args.cache_rate,
            "amp": True,
            "num_gpus": torch.cuda.device_count(),
        },
    )

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
    )
    print(f"[train] device={device} | arch={args.arch} | train batches={len(train_loader)} | val volumes={len(val_loader)}")

    model = build_model(args.arch, device)
    out_ckpt = ckpt_path(args.arch)
    loss_fn = DiceLoss(sigmoid=True, smooth_nr=0.0, smooth_dr=1e-5, squared_pred=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    # Post-processing for validation predictions: sigmoid -> threshold 0.5.
    post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    best_dice = -1.0
    epochs = 1 if args.smoke else args.epochs

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if step % 10 == 0 or args.smoke:
                print(f"  epoch {epoch} step {step}/{len(train_loader)}  loss={loss.item():.4f}")
            if args.smoke and step >= 2:
                break

        scheduler.step()
        mean_loss = epoch_loss / step
        print(f"[train] epoch {epoch} mean loss={mean_loss:.4f}  ({time.time() - t0:.0f}s)")
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": mean_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch_time_s": time.time() - t0,
            },
            step=epoch,
        )

        do_val = args.smoke or epoch % args.val_interval == 0 or epoch == epochs
        if not do_val:
            continue

        # ---- Validation (sliding-window over full volumes) ----
        model.eval()
        dice_metric.reset()
        dice_metric_batch.reset()
        with torch.no_grad():
            for vstep, batch in enumerate(val_loader, 1):
                inputs = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = sliding_window_inference(
                        inputs, ROI_SIZE, args.sw_batch, model, overlap=0.5
                    )
                preds = [post(p) for p in decollate_batch(logits)]
                gts = decollate_batch(labels)
                dice_metric(y_pred=preds, y=gts)
                dice_metric_batch(y_pred=preds, y=gts)
                if args.smoke and vstep >= 2:
                    break

        mean_dice = dice_metric.aggregate().item()
        per_region = dice_metric_batch.aggregate()
        region_str = "  ".join(f"{r}={per_region[i].item():.4f}" for i, r in enumerate(REGIONS))
        print(f"[val]   epoch {epoch}  mean Dice={mean_dice:.4f}  |  {region_str}")
        wandb.log(
            {
                "val/dice_mean": mean_dice,
                **{f"val/dice_{r}": per_region[i].item() for i, r in enumerate(REGIONS)},
            },
            step=epoch,
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({"epoch": epoch, "dice": best_dice, "model": state, "arch": args.arch}, out_ckpt)
            print(f"[val]   new best Dice={best_dice:.4f} -> saved {out_ckpt}")
            wandb.run.summary["best_dice"] = best_dice
            wandb.run.summary["best_epoch"] = epoch

    print(f"[train] done. best validation Dice={best_dice:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
