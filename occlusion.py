"""Occlusion sensitivity for the segmentation model.

Slides a zeroed window over the input and measures how much the predicted
tumour "mass" (summed sigmoid probability for a channel) drops. Regions whose
occlusion collapses the prediction are the ones the model relies on — a
perturbation-based explanation that, unlike Grad-CAM, is well-suited to
segmentation networks. Returns a heatmap in [0, 1].

For speed the perturbation loop runs on a downsampled volume (SegResNet is
fully convolutional); the resulting map is upsampled back to the input size.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _pad_to(x, k=16):
    H, W, D = x.shape[2:]
    ph, pw, pd = (-H) % k, (-W) % k, (-D) % k
    return F.pad(x, (0, pd, 0, pw, 0, ph)), (H, W, D)


@torch.no_grad()
def occlusion_map(model, img, channel=1, window=20, stride=16, downsample=2):
    """(H,W,D) importance map for the given output channel. Higher = more important."""
    orig_size = tuple(img.shape[2:])
    x = img.as_tensor() if hasattr(img, "as_tensor") else img
    if downsample > 1:
        x = F.interpolate(x, scale_factor=1.0 / downsample, mode="trilinear", align_corners=False)

    imgp, (H, W, D) = _pad_to(x, 16)

    def score(t):
        with torch.amp.autocast("cuda"):
            return torch.sigmoid(model(t))[0, channel].sum().item()

    base = score(imgp)
    heat = np.zeros((imgp.shape[2], imgp.shape[3], imgp.shape[4]), np.float32)
    cnt = np.zeros_like(heat)

    brain = (x[0, 0] != 0).cpu().numpy()  # occlude only within the brain box
    if brain.any():
        lo, hi = np.argwhere(brain).min(0), np.argwhere(brain).max(0) + 1
    else:
        lo, hi = (0, 0, 0), (H, W, D)

    for a in range(lo[0], hi[0], stride):
        for b in range(lo[1], hi[1], stride):
            for c in range(lo[2], hi[2], stride):
                occ = imgp.clone()
                occ[:, :, a:a + window, b:b + window, c:c + window] = 0
                drop = max(base - score(occ), 0.0)
                heat[a:a + window, b:b + window, c:c + window] += drop
                cnt[a:a + window, b:b + window, c:c + window] += 1

    cnt[cnt == 0] = 1
    heat = (heat / cnt)[:H, :W, :D]
    # upsample back to the original input resolution
    heat_t = torch.from_numpy(heat)[None, None]
    heat = F.interpolate(heat_t, size=orig_size, mode="trilinear", align_corners=False)[0, 0].numpy()
    if heat.max() > 0:
        heat /= heat.max()
    return heat
