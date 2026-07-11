"""Transforms + DataLoaders for MSD Task01_BrainTumour (BraTS).

Images are 4-channel MRI (FLAIR, T1w, T1gd, T2w). Labels are converted to the
3 standard overlapping BraTS regions:
    TC = Tumor Core        (labels 2 or 3)
    WT = Whole Tumor       (labels 1, 2 or 3)
    ET = Enhancing Tumor   (label  2)
"""

from pathlib import Path

import torch

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch  # noqa: F401 (decollate handy downstream)
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    SpatialPadd,
)

ROOT_DIR = Path("./data")
TASK = "Task01_BrainTumour"


class ConvertDecathlonBratsLabelsd(MapTransform):
    """Convert MSD Task01 labels to the 3 overlapping BraTS regions.

    Decathlon Task01 uses labels {1: edema, 2: non-enhancing tumor,
    3: enhancing tumour} -- which is DIFFERENT from the original BraTS
    convention (1=core, 2=edema, 4=ET) that MONAI's built-in
    ConvertToMultiChannelBasedOnBratsClassesd assumes. Using the built-in
    here silently produces an all-empty ET channel and a wrong TC channel.

        TC (tumor core)   = non-enhancing | enhancing = labels 2 or 3
        WT (whole tumor)  = edema | non-enh | enhancing = labels 1, 2 or 3
        ET (enhancing)    = enhancing = label 3
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1:  # squeeze a singleton channel
                img = img[0]
            result = [
                (img == 2) | (img == 3),  # TC
                (img == 1) | (img == 2) | (img == 3),  # WT
                (img == 3),  # ET
            ]
            d[key] = torch.stack(result, dim=0).float()
        return d

# 3D patch fed to the network during training. 128^3 fits comfortably on a
# 24 GB RTX 3090; bump toward (224, 224, 144) if you want larger context.
ROI_SIZE = (128, 128, 128)


def get_transforms(roi_size=ROI_SIZE):
    """Return (train_transform, val_transform)."""
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),  # 4D image -> channel-first; label gets its channel below
            EnsureTyped(keys=["image", "label"]),
            ConvertDecathlonBratsLabelsd(keys="label"),  # -> 3 channels (TC, WT, ET)
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            # Guarantee every volume is at least roi_size so the random crop
            # always yields an exact roi_size patch (some brains crop < 128).
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )

    # Validation: no cropping or augmentation — evaluate on full volumes
    # (use sliding-window inference at eval time to handle the large size).
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertDecathlonBratsLabelsd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    return train_transform, val_transform


def get_dataloaders(
    batch_size=2,
    num_workers=4,
    roi_size=ROI_SIZE,
    cache_rate=0.0,
):
    """Build train/val DataLoaders from the already-downloaded dataset.

    cache_rate: fraction of each dataset pre-loaded into RAM (MONAI CacheDataset).
                0.0 = stream from disk (low RAM). Raise if you have spare memory.
    """
    train_transform, val_transform = get_transforms(roi_size)

    train_ds = DecathlonDataset(
        root_dir=str(ROOT_DIR),
        task=TASK,
        transform=train_transform,
        section="training",
        download=False,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_ds = DecathlonDataset(
        root_dir=str(ROOT_DIR),
        task=TASK,
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # Full-volume validation -> batch_size 1 (volumes differ in size).
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    import torch

    train_loader, val_loader = get_dataloaders(batch_size=2, num_workers=2)
    print(f"[pipeline] train batches: {len(train_loader)} | val volumes: {len(val_loader)}")

    batch = next(iter(train_loader))
    img, lbl = batch["image"], batch["label"]
    print(f"[pipeline] train image batch: {tuple(img.shape)}  dtype={img.dtype}")
    print(f"[pipeline] train label batch: {tuple(lbl.shape)}  dtype={lbl.dtype}")
    print(f"[pipeline] image value range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"[pipeline] label channels (TC/WT/ET) present: {[int(lbl[:, c].sum() > 0) for c in range(lbl.shape[1])]}")
    print(f"[pipeline] CUDA devices available: {torch.cuda.device_count()}")
