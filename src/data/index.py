"""Build an index (DataFrame) of extracted NEU-CLS images with their class label.

Class is parsed from the filename, e.g. `pitted_surface_123.jpg` -> class
`pitted_surface`, `crazing_7.jpg` -> class `crazing`. This is the only
labeling signal we use — the YOLO bbox .txt files in the source zip are
detection annotations and are not needed for classification.
"""
import re
from pathlib import Path

import pandas as pd

FILENAME_PATTERN = re.compile(r"^(?P<cls>.+)_(?P<idx>\d+)$")


def parse_class_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    match = FILENAME_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Cannot parse class from filename: {filename!r}")
    return match.group("cls")


def build_image_index(raw_dir: Path, valid_extensions=(".jpg",)) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    rows = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() not in valid_extensions:
            continue
        cls = parse_class_from_filename(path.name)
        rows.append({"filename": path.name, "filepath": str(path), "class": cls})
    return pd.DataFrame(rows, columns=["filename", "filepath", "class"])
