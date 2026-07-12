"""Shared inference helpers for the Streamlit demo (Phase 5).

Centralizes: MRI preprocessing, U-Net segmentation (sliding window), feature
extraction, Grad-CAM, and survival prediction -- so app.py stays thin.
"""

from pathlib import Path

import joblib
import numpy as np
import torch

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    NormalizeIntensityd, Orientationd, Spacingd,
)

from data_pipeline import ROI_SIZE
from extract_features import region_features
from occlusion import occlusion_map
from seg_model import BEST_CKPT, load_seg_model

MODALITIES = ["flair", "t1", "t1ce", "t2"]
CKPT = BEST_CKPT
SURV = "survival_model.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_seg_model():
    model, ckpt = load_seg_model(CKPT, DEVICE)
    return model, float(ckpt.get("dice", float("nan")))


def _tf_geom():
    # load + orient + resample only — NO intensity normalization, so the raw
    # FLAIR is kept for a natural, radiology-style display (black background).
    return Compose([
        LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"), EnsureTyped(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ])


_normalize = NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)


def preprocess(paths):
    """paths: list of 4 modality file paths -> (model input 1x4xHxWxD, raw FLAIR np).

    The returned FLAIR is the *raw* (un-normalised) volume for display; the model
    input is the z-score-normalised version.
    """
    data = _tf_geom()({"image": [str(p) for p in paths]})
    display_flair = np.asarray(data["image"][0]).copy()          # raw, for display
    img = _normalize(dict(data))["image"].unsqueeze(0).to(DEVICE)  # model input
    return img, display_flair


def segment(model, img):
    """Return predicted masks (3,H,W,D) bool: TC, WT, ET."""
    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits = sliding_window_inference(img, ROI_SIZE, 4, model, overlap=0.5)
    return (torch.sigmoid(logits)[0] > 0.5).cpu().numpy()


def features(masks):
    """pred_* feature dict from predicted masks (TC,WT,ET)."""
    return region_features(masks[0], masks[1], masks[2], "pred")


def explain(model, img, channel=1):
    """Occlusion-sensitivity heatmap (H,W,D) in [0,1] for the given channel.

    Replaces Grad-CAM, which was measured to not localize on this segmentation
    model (concentration 0.91x, 0% pointing game). Occlusion is perturbation-
    based and suited to segmentation.
    """
    return occlusion_map(model, img, channel=channel)


def load_survival():
    return joblib.load(SURV)


def predict_survival(surv, age, resection_gtr, resection_known, feat):
    """Return (class_idx, prob_vector) from clinical + predicted-mask features."""
    row = {"age": age, "resection_gtr": resection_gtr, "resection_known": resection_known, **feat}
    X = np.array([[row.get(c, 0.0) for c in surv["feature_cols"]]])
    proba = surv["model"].predict_proba(X)[0]
    return int(proba.argmax()), proba
