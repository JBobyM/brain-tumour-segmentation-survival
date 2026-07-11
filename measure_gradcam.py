"""Quantify explanation-map localization against ground-truth tumour masks.

Measures the adopted XAI method (occlusion sensitivity on SegResNet) with the
same metrics used to show Grad-CAM fails, so the two are directly comparable:
  energy-in-tumour : share of heatmap energy inside the true tumour
  volume fraction  : tumour voxels / brain voxels (random baseline)
  concentration    : energy-in-tumour / volume-fraction  (>1 = better than chance)
  pointing game    : does the heatmap peak fall inside the tumour? (% of cases)
  inside/outside   : mean heat inside vs outside the tumour
"""
import glob
import numpy as np
import torch
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    NormalizeIntensityd, Orientationd, Spacingd,
)
import inference as I

BR = "data/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
N = 20


def main():
    model, _ = I.get_seg_model()
    tf = Compose([
        LoadImaged(keys=["image", "seg"]),
        EnsureChannelFirstd(keys=["image", "seg"]),
        EnsureTyped(keys=["image", "seg"]),
        Orientationd(keys=["image", "seg"], axcodes="RAS"),
        Spacingd(keys=["image", "seg"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    folders = sorted(glob.glob(f"{BR}/BraTS20_Training_*"))[:N]
    energy, volfrac, hits, io = [], [], [], []
    for f in folders:
        pid = f.split("/")[-1]
        d = tf({"image": [f"{f}/{pid}_{m}.nii" for m in I.MODALITIES], "seg": f"{f}/{pid}_seg.nii"})
        img = d["image"].unsqueeze(0).to(I.DEVICE)
        seg = np.asarray(d["seg"][0])
        wt = (seg == 1) | (seg == 2) | (seg == 4)          # ground-truth whole tumour
        brain = np.asarray(d["image"][0]) != 0
        cam = I.explain(model, img, channel=1)             # occlusion sensitivity, [0,1]

        e = cam[wt].sum() / (cam[brain].sum() + 1e-8)
        v = wt.sum() / (brain.sum() + 1e-8)
        peak = np.unravel_index(np.where(brain, cam, -1).argmax(), cam.shape)
        energy.append(e); volfrac.append(v); hits.append(bool(wt[peak]))
        io.append(cam[wt].mean() / (cam[brain & ~wt].mean() + 1e-8))

    energy, volfrac = np.array(energy), np.array(volfrac)
    print(f"\n== Occlusion-sensitivity localization (whole-tumour, SegResNet, N={len(folders)}) ==")
    print(f"  energy-in-tumour      : {energy.mean():.3f}")
    print(f"  tumour volume fraction: {volfrac.mean():.3f}  (random baseline)")
    print(f"  concentration ratio   : {(energy/volfrac).mean():.2f}x  (>1 = better than chance)")
    print(f"  pointing game (peak in tumour): {100*np.mean(hits):.0f}%")
    print(f"  inside/outside heat   : {np.mean(io):.2f}  (>1 = tumour hotter than rest of brain)")


if __name__ == "__main__":
    main()
