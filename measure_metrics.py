"""Measure clinical + compute metrics for the case-study README.

Per-region voxel-level sensitivity/specificity + Dice on a validation subset,
plus single-volume inference latency and peak VRAM.
"""
import time
import numpy as np
import torch

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from data_pipeline import get_dataloaders, ROI_SIZE

N_CASES = 30
REGIONS = ["TC", "WT", "ET"]


def main():
    dev = torch.device("cuda")
    model = UNet(spatial_dims=3, in_channels=4, out_channels=3,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2, norm="instance").to(dev)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=dev)["model"])
    model.eval()

    _, val_loader = get_dataloaders(batch_size=1, num_workers=4)

    tp = np.zeros(3); fp = np.zeros(3); fn = np.zeros(3); tn = np.zeros(3)
    times = []
    torch.cuda.reset_peak_memory_stats(dev)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= N_CASES:
                break
            x = batch["image"].to(dev); y = batch["label"].to(dev)
            torch.cuda.synchronize(); t0 = time.time()
            with torch.amp.autocast("cuda"):
                logits = sliding_window_inference(x, ROI_SIZE, 4, model, overlap=0.5)
            torch.cuda.synchronize(); times.append(time.time() - t0)
            p = (torch.sigmoid(logits) > 0.5)
            g = y > 0.5
            for c in range(3):
                pc = p[0, c]; gc = g[0, c]
                tp[c] += (pc & gc).sum().item()
                fp[c] += (pc & ~gc).sum().item()
                fn[c] += (~pc & gc).sum().item()
                tn[c] += (~pc & ~gc).sum().item()

    peak_gb = torch.cuda.max_memory_allocated(dev) / 1e9
    print(f"\n== Compute (single 240x240x155 volume, sliding-window, AMP) ==")
    print(f"  inference latency : {np.mean(times):.2f} s/volume (median {np.median(times):.2f})")
    print(f"  peak VRAM         : {peak_gb:.1f} GB")
    print(f"\n== Clinical metrics (voxel-level, N={N_CASES} validation volumes) ==")
    print(f"  {'region':6s} {'Dice':>6s} {'Sensitivity':>12s} {'Specificity':>12s}")
    for c, r in enumerate(REGIONS):
        dice = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + 1e-8)
        sens = tp[c] / (tp[c] + fn[c] + 1e-8)
        spec = tn[c] / (tn[c] + fp[c] + 1e-8)
        print(f"  {r:6s} {dice:6.3f} {sens:12.3f} {spec:12.3f}")


if __name__ == "__main__":
    main()
