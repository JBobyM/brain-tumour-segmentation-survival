"""Shared inference helpers for the Streamlit demo (Phase 5).

Centralizes: MRI preprocessing, U-Net segmentation (sliding window), feature
extraction, Grad-CAM, and survival prediction -- so app.py stays thin.
"""

from pathlib import Path

import joblib
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
from extract_features import region_features
from gradcam import TARGET_CHANNELS, find_target_layer

MODALITIES = ["flair", "t1", "t1ce", "t2"]
CKPT = "checkpoints/best_model.pth"
SURV = "survival_model.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_seg_model():
    model = UNet(
        spatial_dims=3, in_channels=4, out_channels=3,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, norm="instance",
    ).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, float(ckpt.get("dice", float("nan")))


def _tf():
    return Compose([
        LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"), EnsureTyped(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])


def preprocess(paths):
    """paths: list of 4 modality file paths -> (img tensor 1x4xHxWxD, flair np)."""
    data = _tf()({"image": [str(p) for p in paths]})
    img = data["image"].unsqueeze(0).to(DEVICE)
    return img, img[0, 0].detach().cpu().numpy()


def segment(model, img):
    """Return predicted masks (3,H,W,D) bool: TC, WT, ET."""
    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits = sliding_window_inference(img, ROI_SIZE, 4, model, overlap=0.5)
    return (torch.sigmoid(logits)[0] > 0.5).cpu().numpy()


def features(masks):
    """pred_* feature dict from predicted masks (TC,WT,ET)."""
    return region_features(masks[0], masks[1], masks[2], "pred")


def gradcam(model, img, channel=1):
    """Grad-CAM heatmap (H,W,D) in [0,1] for the given output channel."""
    cam = GradCAM(nn_module=model, target_layers=find_target_layer(model, TARGET_CHANNELS))
    H, W, D = img.shape[2:]
    ph, pw, pd = (-H) % 16, (-W) % 16, (-D) % 16
    img_pad = torch.nn.functional.pad(img, (0, pd, 0, pw, 0, ph))
    cm = cam(x=img_pad, class_idx=channel)[0, 0, :H, :W, :D]
    return cm.detach().cpu().numpy()


def load_survival():
    return joblib.load(SURV)


def predict_survival(surv, age, resection_gtr, resection_known, feat):
    """Return (class_idx, prob_vector) from clinical + predicted-mask features."""
    row = {"age": age, "resection_gtr": resection_gtr, "resection_known": resection_known, **feat}
    X = np.array([[row.get(c, 0.0) for c in surv["feature_cols"]]])
    proba = surv["model"].predict_proba(X)[0]
    return int(proba.argmax()), proba
