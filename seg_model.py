"""Single source of truth for building/loading the segmentation model.

The checkpoint stores its `arch`, so the whole pipeline loads the right
architecture without hard-coding it. SegResNet is the adopted model
(val Dice 0.83 vs U-Net 0.71); the U-Net checkpoint remains loadable as a
baseline (legacy checkpoints without an `arch` field default to unet).
"""
import torch
from monai.networks.nets import SegResNet, UNet

BEST_CKPT = "checkpoints/best_model_segresnet.pth"   # adopted model


def build_seg_model(arch, device):
    if arch == "unet":
        model = UNet(
            spatial_dims=3, in_channels=4, out_channels=3,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
            num_res_units=2, norm="instance",
        )
    elif arch == "segresnet":
        model = SegResNet(
            spatial_dims=3, in_channels=4, out_channels=3,
            init_filters=32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        )
    else:
        raise ValueError(f"unknown arch: {arch}")
    return model.to(device)


def load_seg_model(ckpt_path=BEST_CKPT, device=None):
    """Return (model.eval(), checkpoint_dict). Reads arch from the checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "unet")   # legacy unet checkpoints have no arch field
    model = build_seg_model(arch, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
