"""Phase 4 — Grad-CAM explainability for the segmentation U-Net.

Generates 3D Grad-CAM heatmaps from the encoder bottleneck, showing which
regions drive the whole-tumour prediction, and overlays them on the FLAIR MRI.
This visually audits that the model attends to the tumour (not artefacts) --
addressing the clinical trust requirement in the proposal.

Output: gradcam_overlays.png  (one row per sample case).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    NormalizeIntensityd, Orientationd, Spacingd,
)
from monai.visualize import GradCAM

from data_pipeline import ROI_SIZE

BRATS_DIR = Path("data/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
CKPT = Path("checkpoints/best_model.pth")
OUT_PNG = Path("gradcam_overlays.png")
MODALITIES = ["flair", "t1", "t1ce", "t2"]
# Mid-encoder (128-ch, 16^3) target: localizes better than the 256-ch
# bottleneck, which is too coarse and anti-focuses on segmentation nets.
TARGET_CHANNELS = 128
WT_CHANNEL = 1  # whole-tumour output channel


def find_target_layer(model, out_channels):
    """Name of the last conv in the encoder block with the given width."""
    name = None
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv3d) and m.out_channels == out_channels and "unit1" in n:
            name = n  # take the deepest matching (encoder side)
            break
    return name

# a few representative cases (short / mid / long survivors)
CASES = ["BraTS20_Training_001", "BraTS20_Training_002", "BraTS20_Training_011"]


def load_model(device):
    model = UNet(
        spatial_dims=3, in_channels=4, out_channels=3,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, norm="instance",
    ).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    target_layer = find_target_layer(model, TARGET_CHANNELS)
    print(f"[gradcam] target layer ({TARGET_CHANNELS}ch): {target_layer}")
    cam = GradCAM(nn_module=model, target_layers=target_layer)

    tf = Compose([
        LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"), EnsureTyped(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    fig, axes = plt.subplots(len(CASES), 3, figsize=(12, 4 * len(CASES)))
    if len(CASES) == 1:
        axes = axes[None, :]

    for row, pid in enumerate(CASES):
        folder = BRATS_DIR / pid
        paths = [str(folder / f"{pid}_{m}.nii") for m in MODALITIES]
        img = tf({"image": paths})["image"].unsqueeze(0).to(device)  # (1,4,H,W,D)

        # predicted WT mask (for the tumour-slice pick + contour)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = sliding_window_inference(img, ROI_SIZE, 4, model, overlap=0.5)
        wt_pred = (torch.sigmoid(logits)[0, WT_CHANNEL] > 0.5).cpu().numpy()

        # Grad-CAM on the whole-tumour channel (needs gradients -> no autocast).
        # Pad spatial dims to a multiple of 16 so the U-Net skip-connections
        # concat cleanly on the full volume, then crop the CAM back.
        H, W, D = img.shape[2:]
        ph, pw, pd = (-H) % 16, (-W) % 16, (-D) % 16
        img_pad = torch.nn.functional.pad(img, (0, pd, 0, pw, 0, ph))
        cam_map = cam(x=img_pad, class_idx=WT_CHANNEL)  # (1,1,H',W',D') in [0,1]
        cam_map = cam_map[0, 0, :H, :W, :D].detach().cpu().numpy()

        flair = img[0, 0].detach().cpu().numpy()

        # axial slice with the most predicted tumour (fallback: most CAM)
        z = int(wt_pred.sum((0, 1)).argmax()) if wt_pred.sum() else int(cam_map.sum((0, 1)).argmax())
        base = np.rot90(flair[:, :, z])
        heat = np.rot90(cam_map[:, :, z])
        mask = np.rot90(wt_pred[:, :, z])
        # restrict the heatmap to brain tissue (FLAIR is z-scored with bg=0),
        # removing the coarse-CAM / padding glow outside the skull
        brain = base != 0
        heat_masked = np.ma.masked_where(~brain, heat)
        # per-case full-range contrast within brain so relative hot-spots
        # stand out in every case (raw CAM saturates high on a seg net)
        vals = heat[brain]
        vmin, vmax = (float(vals.min()), np.percentile(vals, 99.5)) if vals.size else (0, 1)

        axes[row, 0].imshow(base, cmap="gray"); axes[row, 0].set_title(f"{pid}\nFLAIR (z={z})")
        axes[row, 1].imshow(base, cmap="gray")
        axes[row, 1].imshow(heat_masked, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
        axes[row, 1].set_title("Grad-CAM (whole-tumour)")
        axes[row, 2].imshow(base, cmap="gray")
        axes[row, 2].contour(mask, levels=[0.5], colors="lime", linewidths=1.5)
        axes[row, 2].set_title("Predicted WT contour")
        for c in range(3):
            axes[row, c].axis("off")
        print(f"[gradcam] {pid}: slice z={z}, CAM range [{cam_map.min():.2f},{cam_map.max():.2f}], "
              f"WT voxels={int(wt_pred.sum())}")

    fig.suptitle("Grad-CAM — where the segmentation model looks for whole tumour", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    print(f"[gradcam] saved -> {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
