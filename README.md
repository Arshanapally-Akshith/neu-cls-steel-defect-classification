# NEU-CLS Steel Defect Classification

6-class steel surface defect classification via transfer learning, with
rigorous error analysis and Grad-CAM verification, deployed as a Streamlit
demo. See [WORKFLOW.md](<WORKFLOW%20(3).md>) for the full phase-by-phase plan.

**Status:** Phases 1–6 complete (data prep, classical baseline, ResNet18
transfer learning, error analysis, Grad-CAM verification, Streamlit demo).
Phase 7 (write-up) not yet done.

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
  images and frozen under `data/splits/`. One exact-duplicate pair was found
  and resolved (see `reports/data_integrity.md`), leaving 1,799 unique images.

## Setup

```bash
pip install -r requirements.txt
# PyTorch is CPU-only in requirements.txt; for a CUDA build instead, see:
# https://pytorch.org/get-started/locally/
```

## Running the pipeline (in order)

Each phase's script is idempotent and reads only the frozen artifacts the
previous phase produced — never re-derives them.

```bash
python -m scripts.run_phase1               # data integrity + frozen 70/15/15 split
python -m scripts.run_phase2_baseline       # HOG + Logistic Regression baseline
python -m scripts.run_phase3_transfer       # ResNet18 transfer learning (~30 min on CPU)
python -m scripts.run_phase4_error_analysis # per-sample error analysis vs. both models
python -m scripts.run_phase5_gradcam        # Grad-CAM heatmaps + honest attention analysis
```

Each writes its report(s) under `reports/` (see that directory for the
full list) and, where applicable, a model artifact under `models/`. Those
are gitignored and regenerable via the scripts above — except
`models/resnet18_finetuned.pt`, which is committed (see the Streamlit
section below for why).

## Streamlit Demo (Phase 6)

```bash
streamlit run streamlit_app.py
```

Upload a steel-surface image and the app shows the predicted defect class,
its confidence, a Grad-CAM overlay, and (if applicable) a note when the
prediction falls into Phase 4's known confusion group
(`inclusion`/`pitted_surface`/`scratches`). Requires
`models/resnet18_finetuned.pt` to exist — it's committed to this repo (see
below), so a fresh clone works with no retraining step. If it's ever
missing, regenerate it with `python -m scripts.run_phase3_transfer`
(~30 min on CPU).

The app is a thin UI layer only: all model loading, preprocessing,
prediction, and Grad-CAM logic is reused from `src/models/transfer.py` and
`src/gradcam/gradcam.py`, not reimplemented.

**Deploying to Streamlit Community Cloud:** point it at `streamlit_app.py`
on this repo — no extra setup needed. `models/resnet18_finetuned.pt`
(~45MB) is committed as a deliberate, explicit exception to the general
"model artifacts are gitignored" policy (see `.gitignore`), specifically so
the demo runs immediately on a fresh clone/deploy without a ~30 min CPU
retraining step first. Every other model artifact (`models/baseline.joblib`,
etc.) stays gitignored and regenerable, as before. There are no secrets to
configure.

## Tests

```bash
pytest
```

Covers: dataset integrity, split correctness/reproducibility, the HOG
baseline pipeline (including no train/val/test leakage), the ResNet18
training/CV/evaluation pipeline (including a regression test that the
frozen test manifest is never read during cross-validation), Phase 4's
error-analysis utilities (including a bug-regression test), Phase 5's
Grad-CAM implementation (including a real-checkpoint, read-only
integration test), and Phase 6's single-image prediction helper plus
Streamlit-app smoke tests (via `streamlit.testing.v1.AppTest`).

## Repo Structure

```
├── streamlit_app.py             # Phase 6 demo (thin UI layer, reuses src/)
├── requirements.txt
├── config/config.yaml           # every phase's settings, one file
├── src/
│   ├── config.py                 # config.yaml loader
│   ├── data/                     # extraction, indexing, integrity, splitting, loading
│   ├── features/                 # HOG feature extraction (Phase 2)
│   ├── models/                   # baseline.py (Phase 2), transfer.py (Phase 3, ResNet18)
│   ├── eval/                     # shared metrics, confusion matrix, error analysis
│   └── gradcam/                  # Grad-CAM, image selection, visualization (Phase 5)
├── scripts/                     # one orchestrator script per phase
├── data/
│   ├── raw/                      # extracted images (gitignored)
│   └── splits/                   # frozen split manifests (committed)
├── models/                      # trained model artifacts — gitignored/regenerable,
│                                 #   except resnet18_finetuned.pt (committed for the demo)
├── reports/                     # every phase's report + figures (committed)
└── tests/                       # one test module per component
```
