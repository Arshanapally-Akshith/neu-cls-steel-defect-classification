# Data Integrity Report — NEU-CLS

Generated: 2026-08-08

**Note on the source archive:** the provided `NEU-CLS.zip` contains image files *alongside* object-detection annotations (YOLO-format bounding-box `.txt` labels, laid out as `train/train/images/` + `train/train/labels/`, `valid/valid/images/` + `valid/valid/labels/`). This project uses **only the images**, for 6-class whole-image classification, and **intentionally ignores the detection labels** — no bounding-box information is used anywhere in this pipeline. Only the 1,800 `.jpg` images were extracted; the zip's own train(295)/valid(5) per-class split is a detection split and was discarded. Class is parsed from the filename prefix (e.g. `crazing_10.jpg` -> `crazing`).

## Class List

- `crazing`
- `inclusion`
- `patches`
- `pitted_surface`
- `rolled-in_scale`
- `scratches`

## Dataset Counts

- Raw files extracted: **1800**
- Total images indexed: **1800** (expected 1800) — **PASS**

| Class | Count | Expected |
|---|---|---|
| crazing | 300 | 300 |
| inclusion | 300 | 300 |
| patches | 300 | 300 |
| pitted_surface | 300 | 300 |
| rolled-in_scale | 300 | 300 |
| scratches | 300 | 300 |

- Class balance: **PASS**

## Corruption Check

- Corrupted files found: **0**

## Dimensions & Channel Mode

- Expected size: [200, 200] (width x height)
- Expected mode: `RGB`
- Size distribution: {'(200, 200)': 1800}
- Mode distribution: {'RGB': 1800}
- All images uniform size: **PASS**
- All images uniform mode: **PASS**

## Exact Duplicate Check (SHA-256)

- Unique file hashes: 1799
- Duplicate groups found: 1
- Duplicate files (excess copies): 1
- Has duplicates: **FAIL** (PASS = no duplicates)
  - ['patches_101.jpg', 'patches_105.jpg']

## Deduplication (Remediation)

The exact-duplicate check above found 1 duplicate file(s). Byte-identical images must never land in different splits (that would leak the same image between, e.g., train and val), so before building the train/val/test split, one copy of each duplicate group was dropped (keeping the alphabetically-first filename):
  - dropped `patches_105.jpg`

Split is therefore built from **1799** unique images, not the raw 1800.

## Overall Integrity Result

Raw data, before deduplication: **FAIL** (counts/corruption/dimensions all pass; duplicate check is the one finding — see above, resolved via deduplication before splitting).

## Train/Val/Test Split

Fixed stratified split, seed=`42`, fractions=train=0.7, val=0.15, test=0.15, built from the post-deduplication image pool. Manifests are frozen CSV files under `data/splits/` (filename + class only, no absolute paths) — every later phase must load these same files rather than re-splitting.

| Split | Total | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
|---|---|---|---|---|---|---|---|
| train | 1259 | 210 | 210 | 209 | 210 | 210 | 210 |
| val | 270 | 45 | 45 | 45 | 45 | 45 | 45 |
| test | 270 | 45 | 45 | 45 | 45 | 45 | 45 |

## Early Visual Notes

- `rolled-in_scale` and `crazing` are the two classes most often flagged in the NEU-CLS literature as visually similar (both present as diffuse, low-contrast linear/mottled texture rather than a sharp localized defect like `scratches` or `patches`). This is a hypothesis to verify empirically in Phase 4 (confusion matrix), not a confirmed finding from Phase 1.
- All images are uniform 200x200 and stored as JPEG in RGB mode despite the underlying content being grayscale (see Dimensions & Channel Mode above) — worth keeping in mind for Phase 3 preprocessing (no channel-count mismatch to handle, but redundant channels).
- The source zip is a detection-annotation release (YOLO bbox labels per image, 295/5 train/valid split). This project only needs classification labels, so bbox labels were ignored and a fresh 70/15/15 stratified split was built from the pooled 1,800 images.
