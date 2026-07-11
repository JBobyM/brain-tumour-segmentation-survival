"""Phase 1 sanity check: visualize normalized slices from a sample BraTS volume.

Saves a PNG grid (4 MRI modalities + 3 label regions) so the pipeline output
can be eyeballed before training. Runs headless (no display needed).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
)

from data_pipeline import ConvertDecathlonBratsLabelsd

ROOT_DIR = Path("./data")
TASK = "Task01_BrainTumour"
OUT_PNG = Path("sample_slice.png")

MODALITIES = ["FLAIR", "T1w", "T1gd", "T2w"]
REGIONS = ["TC (tumor core)", "WT (whole tumor)", "ET (enhancing)"]


def main(sample_idx: int = 0):
    # Minimal transform: load, normalize, convert labels — no cropping/aug.
    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertDecathlonBratsLabelsd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    ds = DecathlonDataset(
        root_dir=str(ROOT_DIR),
        task=TASK,
        transform=transform,
        section="training",
        download=False,
        cache_rate=0.0,
    )

    sample = ds[sample_idx]
    img = np.asarray(sample["image"])  # (4, H, W, D)
    lbl = np.asarray(sample["label"])  # (3, H, W, D)

    print(f"[verify] image shape: {img.shape}  label shape: {lbl.shape}")
    print(f"[verify] per-modality intensity (mean/std after norm):")
    for c, name in enumerate(MODALITIES):
        print(f"          {name:5s}: mean={img[c].mean():+.3f} std={img[c].std():.3f}")

    # Choose the axial slice with the most tumour (whole-tumour channel = idx 1).
    wt = lbl[1]
    slice_idx = int(wt.sum(axis=(0, 1)).argmax()) if wt.sum() > 0 else img.shape[-1] // 2
    print(f"[verify] plotting axial slice z={slice_idx} (max tumour extent)")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: the 4 MRI modalities
    for c, name in enumerate(MODALITIES):
        axes[0, c].imshow(np.rot90(img[c, :, :, slice_idx]), cmap="gray")
        axes[0, c].set_title(name)
        axes[0, c].axis("off")

    # Row 2: FLAIR with each label region overlaid, + combined
    base = np.rot90(img[0, :, :, slice_idx])
    colors = ["Reds", "Greens", "Blues"]
    for c, (name, cmap) in enumerate(zip(REGIONS, colors)):
        axes[1, c].imshow(base, cmap="gray")
        mask = np.rot90(lbl[c, :, :, slice_idx])
        axes[1, c].imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        axes[1, c].set_title(f"{name}\n{int(mask.sum())} px")
        axes[1, c].axis("off")

    # Combined RGB overlay in the last cell
    axes[1, 3].imshow(base, cmap="gray")
    rgb = np.stack([np.rot90(lbl[i, :, :, slice_idx]) for i in range(3)], axis=-1)
    axes[1, 3].imshow(rgb, alpha=0.5)
    axes[1, 3].set_title("TC=R  WT=G  ET=B")
    axes[1, 3].axis("off")

    fig.suptitle(f"{TASK} — sample {sample_idx}, axial slice {slice_idx}", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    print(f"[verify] saved figure -> {OUT_PNG.resolve()}")


if __name__ == "__main__":
    import sys

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(idx)
