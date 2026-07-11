"""Phase 3, step 1 — extract survival features from BraTS 2020.

For each patient with survival info we derive interpretable, clinically
motivated features from the tumour segmentation, plus clinical covariates
(age, resection). Two feature sets are produced:

  * predicted masks  -> from the Phase 2 U-Net run end-to-end on the MRI
  * ground-truth masks -> from the BraTS expert segmentation (upper bound)

Output: survival_features.csv  (one row per patient, both feature sets +
clinical + 3-class survival label).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import nibabel as nib

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

from data_pipeline import ROI_SIZE

BRATS_DIR = Path("data/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
CKPT = Path("checkpoints/best_model.pth")
OUT_CSV = Path("survival_features.csv")
MODALITIES = ["flair", "t1", "t1ce", "t2"]  # model's trained channel order


def survival_class(days):
    if pd.isna(days):
        return np.nan
    if days < 300:
        return 0  # short  (<10 mo)
    if days <= 450:
        return 1  # mid    (10-15 mo)
    return 2      # long   (>15 mo)


def region_features(tc, wt, et, prefix):
    """Interpretable features from binary TC/WT/ET masks (1mm iso -> mm^3)."""
    tc_v, wt_v, et_v = float(tc.sum()), float(wt.sum()), float(et.sum())
    necrotic = max(tc_v - et_v, 0.0)   # non-enhancing core
    edema = max(wt_v - tc_v, 0.0)      # peritumoral edema
    eps = 1.0
    f = {
        f"{prefix}_vol_tc": tc_v,
        f"{prefix}_vol_wt": wt_v,
        f"{prefix}_vol_et": et_v,
        f"{prefix}_vol_necrotic": necrotic,
        f"{prefix}_vol_edema": edema,
        f"{prefix}_ratio_et_wt": et_v / (wt_v + eps),
        f"{prefix}_ratio_tc_wt": tc_v / (wt_v + eps),
        f"{prefix}_ratio_et_tc": et_v / (tc_v + eps),
        f"{prefix}_ratio_necrotic_tc": necrotic / (tc_v + eps),
    }
    # whole-tumour shape: bounding-box extent + compactness
    idx = np.argwhere(wt > 0)
    if len(idx):
        mins, maxs = idx.min(0), idx.max(0) + 1
        dx, dy, dz = (maxs - mins).astype(float)
        bbox_vol = dx * dy * dz
        f[f"{prefix}_wt_bbox_dx"] = dx
        f[f"{prefix}_wt_bbox_dy"] = dy
        f[f"{prefix}_wt_bbox_dz"] = dz
        f[f"{prefix}_wt_compactness"] = wt_v / (bbox_vol + eps)
    else:
        for k in ("dx", "dy", "dz"):
            f[f"{prefix}_wt_bbox_{k}"] = 0.0
        f[f"{prefix}_wt_compactness"] = 0.0
    return f


def load_model(device):
    model = UNet(
        spatial_dims=3, in_channels=4, out_channels=3,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, norm="instance",
    ).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[extract] loaded Phase 2 model (val Dice={ckpt.get('dice'):.4f}, epoch {ckpt.get('epoch')})")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    img_tf = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    df = pd.read_csv(BRATS_DIR / "survival_info.csv")
    rows = []
    n = len(df)
    for i, r in df.iterrows():
        pid = r["Brats20ID"]
        folder = BRATS_DIR / pid
        try:
            paths = [str(folder / f"{pid}_{m}.nii") for m in MODALITIES]
            data = img_tf({"image": paths})
            img = data["image"].unsqueeze(0).to(device)  # (1,4,H,W,D)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                logits = sliding_window_inference(img, ROI_SIZE, 4, model, overlap=0.5)
            pred = (torch.sigmoid(logits)[0] > 0.5).cpu().numpy()  # (3,H,W,D): TC,WT,ET
            feat_pred = region_features(pred[0], pred[1], pred[2], "pred")

            # ground-truth masks (BraTS labels 1/2/4)
            seg = np.asarray(nib.load(folder / f"{pid}_seg.nii").dataobj)
            gt_tc = (seg == 1) | (seg == 4)
            gt_wt = (seg == 1) | (seg == 2) | (seg == 4)
            gt_et = (seg == 4)
            feat_gt = region_features(gt_tc, gt_wt, gt_et, "gt")

            age = float(r["Age"])
            res = str(r["Extent_of_Resection"]).upper()
            row = {
                "Brats20ID": pid,
                "age": age,
                "resection_gtr": 1 if "GTR" in res else 0,
                "resection_known": 0 if res in ("NAN", "NA", "") else 1,
                "survival_days": pd.to_numeric(r["Survival_days"], errors="coerce"),
                "survival_class": survival_class(pd.to_numeric(r["Survival_days"], errors="coerce")),
                **feat_pred, **feat_gt,
            }
            rows.append(row)
        except Exception as e:
            print(f"[extract] WARN {pid}: {e}")
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"[extract] {i + 1}/{n} cases done")
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)  # incremental save

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[extract] wrote {len(out)} rows x {out.shape[1]} cols -> {OUT_CSV}")
    print("[extract] class distribution:", out["survival_class"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
