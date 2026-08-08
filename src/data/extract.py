"""Extract the NEU-CLS dataset zip into data/raw/ (idempotent).

The source zip ships in a detection-annotation layout:

    train/train/images/<class>_<n>.jpg   (295 per class)
    train/train/labels/<class>_<n>.txt   (YOLO bbox labels, unused here)
    valid/valid/images/<class>_<n>.jpg   (5 per class)
    valid/valid/labels/<class>_<n>.txt

We only need the 1,800 images for classification, so extraction flattens
both image subtrees into a single directory and ignores the label files.
The zip's own train/valid split is a detection split (295/5), not a
classification split, and is discarded — Phase 1 builds its own fixed
stratified 70/15/15 split from the pooled 1,800 images.
"""
import shutil
import zipfile
from pathlib import Path

IMAGE_DIR_MARKER = "/images/"


def extract_dataset(zip_path: Path, raw_dir: Path, force: bool = False) -> Path:
    """Extract images from zip_path into raw_dir (flat, one file per image).

    Idempotent: if raw_dir already exists and looks populated, does nothing
    unless force=True.
    """
    zip_path = Path(zip_path)
    raw_dir = Path(raw_dir)

    if raw_dir.exists() and force:
        shutil.rmtree(raw_dir)

    if raw_dir.exists() and any(raw_dir.iterdir()):
        return raw_dir

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset zip not found at {zip_path}. Expected it to already be "
            "present in the repository."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        image_entries = [
            n for n in zf.namelist() if IMAGE_DIR_MARKER in n and not n.endswith("/")
        ]
        for entry in image_entries:
            filename = Path(entry).name
            dest = raw_dir / filename
            with zf.open(entry) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    return raw_dir


if __name__ == "__main__":
    from src.config import load_config, resolve_path

    config = load_config()
    zip_path = resolve_path(config["dataset"]["zip_path"])
    raw_dir = resolve_path(config["dataset"]["raw_dir"])
    extracted = extract_dataset(zip_path, raw_dir)
    n_files = len(list(extracted.iterdir()))
    print(f"Extracted {n_files} files into {extracted}")
