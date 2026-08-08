# NEU-CLS Steel Defect Classification

6-class steel surface defect classification via transfer learning, with
rigorous error analysis and Grad-CAM verification, deployed as a Streamlit
demo. See [WORKFLOW.md](<WORKFLOW%20(3).md>) for the full phase-by-phase plan.

**Status:** Phase 1 (Data Prep & Integrity) complete. Phases 2+ (baseline,
transfer learning, error analysis, Grad-CAM, Streamlit) not yet implemented.

## Dataset

NEU-CLS: 1,800 grayscale-content 200x200 images, 6 classes, 300 images per
class (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`,
`scratches`).

**Note on the source archive:** the provided `NEU-CLS.zip` (not committed to
git — see `.gitignore`) contains image files *alongside* object-detection
annotations (YOLO-format bounding-box `.txt` labels, laid out as
`train/train/images/` + `train/train/labels/`, `valid/valid/images/` +
`valid/valid/labels/`). This project uses **only the images**, for 6-class
whole-image classification, and **intentionally ignores the detection
labels** — no bounding-box information is used anywhere in this pipeline.
Concretely:

- Only the `.jpg` images are extracted; the `.txt` bounding-box label files are never read.
- The zip's own train(295)/valid(5) per-class split is a detection split, not
  a classification split — it is discarded.
- A fresh, fixed, stratified 70/15/15 split is built from the pooled 1,800
  images and frozen under `data/splits/`.

## Setup

```bash
pip install -r requirements.txt
```

## Running Phase 1

```bash
python -m scripts.run_phase1
```

This extracts `NEU-CLS.zip` into `data/raw/NEU-CLS/` (gitignored), runs all
integrity checks, builds the fixed stratified split, writes the split
manifests to `data/splits/`, and writes `reports/data_integrity.md` +
`reports/data_integrity.json`.

## Tests

```bash
pytest
```

Covers: dataset integrity (counts, class balance, corruption, dimensions,
exact duplicates) and the train/val/test split (correct proportions,
disjointness, full coverage, per-class balance, reproducibility under a fixed
seed).

## Repo Structure

```
├── streamlit_app.py            # Phase 6 (not yet implemented)
├── requirements.txt
├── config/config.yaml          # fixed split seed, class list, paths
├── src/
│   ├── config.py                # config.yaml loader
│   └── data/                    # extraction, indexing, integrity checks, splitting
├── scripts/run_phase1.py       # Phase 1 orchestrator
├── data/
│   ├── raw/                     # extracted images (gitignored)
│   └── splits/                  # frozen split manifests (committed)
├── reports/                     # data_integrity.md, etc.
└── tests/                       # integrity + split tests
```
