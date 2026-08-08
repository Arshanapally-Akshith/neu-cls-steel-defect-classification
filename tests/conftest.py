import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.dedup import resolve_duplicates
from src.data.extract import extract_dataset
from src.data.index import build_image_index
from src.data.integrity import check_duplicates


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def raw_dir(config):
    zip_path = resolve_path(config["dataset"]["zip_path"])
    raw_dir = resolve_path(config["dataset"]["raw_dir"])
    if not zip_path.exists():
        pytest.skip(f"Dataset zip not found at {zip_path}; skipping data-dependent tests.")
    extract_dataset(zip_path, raw_dir)
    return raw_dir


@pytest.fixture(scope="session")
def image_index(config, raw_dir):
    return build_image_index(raw_dir, valid_extensions=tuple(config["dataset"]["valid_extensions"]))


@pytest.fixture(scope="session")
def deduped_index(image_index):
    """The image index actually used for splitting: exact duplicates resolved.

    Mirrors what scripts/run_phase1.py does before calling stratified_split —
    see src/data/dedup.py for why (byte-identical images must not be able to
    land in different splits).
    """
    duplicates = check_duplicates(image_index)
    deduped_df, _dropped = resolve_duplicates(image_index, duplicates["duplicate_groups"])
    return deduped_df
