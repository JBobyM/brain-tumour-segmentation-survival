"""Load the MSD Task01_BrainTumour dataset, downloading only if needed."""

from pathlib import Path

from monai.apps import DecathlonDataset

ROOT_DIR = Path("./data")
TASK = "Task01_BrainTumour"


def data_exists(root_dir: Path, task: str) -> bool:
    """True if the extracted dataset (with its dataset.json) is already present."""
    return (root_dir / task / "dataset.json").is_file()


def get_dataset(section: str = "training", cache_rate: float = 0.0, transform=None):
    ROOT_DIR.mkdir(parents=True, exist_ok=True)

    need_download = not data_exists(ROOT_DIR, TASK)
    if need_download:
        print(f"[data] {TASK} not found in {ROOT_DIR} — downloading (~7 GB, first run only)...")
    else:
        print(f"[data] Found existing {TASK} in {ROOT_DIR} — skipping download.")

    return DecathlonDataset(
        root_dir=str(ROOT_DIR),
        task=TASK,
        section=section,
        download=need_download,
        cache_rate=cache_rate,
        transform=transform,  # None => items are dicts with raw "image"/"label" paths
    )


if __name__ == "__main__":
    ds = get_dataset(section="training")
    print(f"[data] Loaded {len(ds)} training samples.")
    sample = ds[0]
    print(f"[data] First sample keys: {list(sample.keys())}")
