"""Predicted-vs-ground-truth figures for the segmentation model.

  pred_vs_gt_segmentation.png : best / median / worst cases (by WT Dice) with
                                GT vs predicted masks and a TP/FN/FP error map,
                                plus a per-region Dice distribution.
  pred_vs_gt_volume.png       : predicted vs expert tumour volumes (TC/WT/ET).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

from monai.inferers import sliding_window_inference
from data_pipeline import get_dataloaders, ROI_SIZE
from seg_model import load_seg_model

N = 24
REGIONS = ["TC", "WT", "ET"]


def dice(p, g):
    tp = (p & g).sum()
    return 2 * tp / (p.sum() + g.sum() + 1e-8)


def segmentation_figure():
    dev = torch.device("cuda")
    model, ck = load_seg_model(device=dev)
    _, val = get_dataloaders(batch_size=1, num_workers=4)

    cases, per_region = [], {r: [] for r in REGIONS}
    with torch.no_grad():
        for i, b in enumerate(val):
            if i >= N:
                break
            x = b["image"].to(dev)
            g = (b["label"][0].numpy() > 0.5)
            with torch.amp.autocast("cuda"):
                logits = sliding_window_inference(x, ROI_SIZE, 4, model, overlap=0.5)
            p = (torch.sigmoid(logits)[0] > 0.5).cpu().numpy()
            for c, r in enumerate(REGIONS):
                per_region[r].append(dice(p[c], g[c]))
            gw, pw = g[1], p[1]                       # whole tumour
            z = int(gw.sum((0, 1)).argmax()) if gw.sum() else gw.shape[-1] // 2
            cases.append(dict(
                wt_dice=dice(pw, gw),
                flair=np.rot90(x[0, 0].cpu().numpy()[:, :, z]),
                gt=np.rot90(gw[:, :, z]), pred=np.rot90(pw[:, :, z]),
            ))

    cases.sort(key=lambda c: c["wt_dice"])
    picks = [("Worst", cases[0]), ("Median", cases[len(cases) // 2]), ("Best", cases[-1])]

    fig, ax = plt.subplots(3, 4, figsize=(15, 11))
    err_cmap = ListedColormap([[0, 0, 0, 0], "#2ca02c", "#d62728", "#ff7f0e"])  # -, TP, FN, FP
    for row, (label, c) in enumerate(picks):
        base, gt, pr = c["flair"], c["gt"], c["pred"]
        ax[row, 0].imshow(base, cmap="gray"); ax[row, 0].set_ylabel(f"{label}\nWT Dice {c['wt_dice']:.2f}", fontsize=11)
        ax[row, 0].set_title("FLAIR" if row == 0 else "");
        ax[row, 1].imshow(base, cmap="gray")
        ax[row, 1].imshow(np.ma.masked_where(~gt, gt), cmap="Greens", alpha=.6, vmin=0, vmax=1)
        ax[row, 1].set_title("Ground truth (WT)" if row == 0 else "")
        ax[row, 2].imshow(base, cmap="gray")
        ax[row, 2].imshow(np.ma.masked_where(~pr, pr), cmap="Reds", alpha=.6, vmin=0, vmax=1)
        ax[row, 2].set_title("Predicted (WT)" if row == 0 else "")
        err = np.zeros_like(gt, dtype=int)
        err[gt & pr] = 1; err[gt & ~pr] = 2; err[~gt & pr] = 3
        ax[row, 3].imshow(base, cmap="gray"); ax[row, 3].imshow(err, cmap=err_cmap, alpha=.7)
        ax[row, 3].set_title("Error map" if row == 0 else "")
        for k in range(4):
            ax[row, k].set_xticks([]); ax[row, k].set_yticks([])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#2ca02c", label="Correct (TP)"),
                        Patch(color="#d62728", label="Missed (FN)"),
                        Patch(color="#ff7f0e", label="Over-seg (FP)")],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Predicted vs ground truth — whole-tumour segmentation (SegResNet, Dice {ck['dice']:.3f})", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig("pred_vs_gt_segmentation.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # per-region Dice distribution
    fig, axb = plt.subplots(figsize=(6, 4.5))
    axb.boxplot([per_region[r] for r in REGIONS], tick_labels=REGIONS, showmeans=True)
    axb.set_ylabel("per-case Dice"); axb.set_title(f"Per-case Dice distribution (N={N} validation)")
    axb.set_ylim(0, 1); axb.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig("pred_vs_gt_dice_dist.png", dpi=110); plt.close(fig)
    print(f"[fig] segmentation: WT Dice range {cases[0]['wt_dice']:.2f}-{cases[-1]['wt_dice']:.2f}; "
          f"means TC/WT/ET = {[round(np.mean(per_region[r]),2) for r in REGIONS]}")


def volume_figure():
    df = pd.read_csv("survival_features.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, r in zip(ax, ["tc", "wt", "et"]):
        p = df[f"pred_vol_{r}"] / 1000; g = df[f"gt_vol_{r}"] / 1000
        a.scatter(g, p, s=14, alpha=.5, color="#4C78A8")
        m = max(g.max(), p.max())
        a.plot([0, m], [0, m], "k--", alpha=.5, label="y = x")
        rr = np.corrcoef(g, p)[0, 1]
        a.set_title(f"{r.upper()}  (r = {rr:.2f})"); a.set_xlabel("expert volume (cm³)")
        a.set_ylabel("predicted volume (cm³)"); a.legend(loc="upper left", fontsize=8)
    fig.suptitle("Predicted vs expert tumour volume — 235 BraTS cases", fontsize=14)
    fig.tight_layout(); fig.savefig("pred_vs_gt_volume.png", dpi=110); plt.close(fig)
    print("[fig] volume scatter saved")


if __name__ == "__main__":
    segmentation_figure()
    volume_figure()
    print("[fig] wrote pred_vs_gt_segmentation.png, pred_vs_gt_dice_dist.png, pred_vs_gt_volume.png")
