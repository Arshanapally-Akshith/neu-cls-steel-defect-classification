"""Resolve exact (byte-for-byte) duplicate images before splitting.

The raw integrity check (integrity.check_duplicates) reports duplicate
groups but does not alter the data — that check exists to describe the raw
dataset honestly. Deduplication is a separate, explicit remediation step:
for each duplicate group, keep the alphabetically-first filename and drop
the rest, so no two splits can end up holding byte-identical images (which
would otherwise leak information from train into val/test).
"""
from pathlib import Path

import pandas as pd


def resolve_duplicates(df: pd.DataFrame, duplicate_groups: list[list[str]]) -> tuple[pd.DataFrame, list[str]]:
    """Drop all but the first (by filename) file in each duplicate group.

    duplicate_groups: list of filepath lists, as returned by
    integrity.check_duplicates()["duplicate_groups"].

    Returns (deduped_df, dropped_filepaths).
    """
    dropped: list[str] = []
    for group in duplicate_groups:
        kept = min(group, key=lambda p: Path(p).name)
        dropped.extend(p for p in group if p != kept)

    deduped_df = df[~df["filepath"].isin(dropped)].reset_index(drop=True)
    return deduped_df, dropped
