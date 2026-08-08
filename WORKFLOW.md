# NEU-CLS Steel Defect Classification — Project Workflow

**Scope:** 6-class steel surface defect classification via transfer learning, with
rigorous error analysis and Grad-CAM verification, deployed as a Streamlit demo.
Single dataset, no scope creep.

**Dataset:** NEU-CLS (Figshare, ~26MB) — 1,800 grayscale 200x200 images, 6 classes
(Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches), 300 images
per class. Source: https://figshare.com/articles/dataset/NEU-CLS/28903550

---

## Phase 1 — Data Prep & Integrity Check

- Download NEU-CLS (classification-only version, no detection annotations needed).
- Verify class balance: confirm 300 images/class x 6 = 1,800 total, no corrupted or
  duplicate files.
- Split: stratified train/val/test (e.g. 70/15/15). **Fix this split once and reuse
  it everywhere** — every later comparison (baseline vs. model, fold vs. fold) must
  use the same held-out data, or the results aren't comparable.
- Document class list, image characteristics, and any early visual notes (which
  classes look similar on inspection) in `reports/data_integrity.md`.

**Deliverable:** `data/processed/` with fixed splits + `reports/data_integrity.md`

---

## Phase 2 — Baseline

- Train a simple baseline on the same fixed split: either a shallow CNN from
  scratch, or a non-DL baseline (HOG features + logistic regression / random
  forest).
- Purpose: establish whether transfer learning *actually* helps, with a real
  number to compare against — not just one accuracy figure reported in isolation.

**Deliverable:** `reports/baseline_results.md`, `models/baseline.*`

---

## Phase 3 — Transfer Learning Model

- Pretrained backbone: MobileNetV2 (lightweight, fast Streamlit inference) or
  ResNet18. Freeze backbone, fine-tune classification head.
- Augmentation: rotation, flip, slight crop/zoom (defects are texture/pattern
  based — these are safe). Skip color jitter; images are grayscale, so it's
  meaningless here.
- Use k-fold cross-validation, not a single train/test run. With only 1,800
  images, a single split has real variance — report mean ± std across folds.

**Deliverable:** `models/finetuned_model.*`, `reports/cv_results.md`

---

## Phase 4 — Error Analysis (the differentiator)

- Full confusion matrix on held-out test data.
- Explicitly discuss which classes get confused (literature flags rolled-in-scale
  vs. crazing as visually similar — check if your model shows the same pattern).
- Report per-class precision/recall, not just overall accuracy. State plainly
  which classes are hardest and why, if you can tell.

**Deliverable:** `reports/error_analysis.md` (confusion matrix + per-class metrics
+ written discussion)

---

## Phase 5 — Grad-CAM Verification

- Generate Grad-CAM heatmaps on both correctly and incorrectly classified examples.
- Check: is the model attending to the actual defect region, or latching onto
  shortcut features (image borders, uniform lighting bands)?
- This is a falsifiable claim — report honestly even if the model sometimes
  attends to the wrong region. That's a legitimate finding, not a failure.

**Deliverable:** `reports/gradcam_analysis.md` + saved heatmap examples

---

## Phase 6 — Streamlit Demo

- App: upload image → predicted class + confidence + Grad-CAM overlay.
- Optional: surface the Phase 4 finding directly in the UI (e.g. flag when the
  model's top-2 predictions are a known confusion-prone pair) — turns the error
  analysis into a demo feature, not just a report table.

**Deliverable:** `streamlit_app.py`, deployed to Streamlit Community Cloud

---

## Phase 7 — Write-Up

- `reports/interview_prep.md`, same style as GridForecast/SegmentChurn.
  Anticipated questions to answer in writing:
  - Why transfer learning over training from scratch?
  - Why k-fold over a single split?
  - What would break if deployed on a different steel mill's camera setup
    (domain shift — state this caveat upfront, don't wait to be asked)?
  - What does the Grad-CAM analysis actually show, and what does it not prove?

---

## Repo Structure (reference)

```
neu-defect-classification/
├── streamlit_app.py
├── requirements.txt
├── config/config.yaml          # fixed split seed, class list, model choice
├── src/
│   ├── data/                   # loading, splitting, integrity checks
│   ├── models/                 # baseline + transfer learning model defs
│   ├── eval/                   # cross-validation, confusion matrix, metrics
│   └── gradcam/                # Grad-CAM generation
├── models/                     # saved model artifacts
├── data/processed/             # fixed splits (small enough to commit)
├── notebooks/                  # EDA, exploration
├── reports/                    # all phase reports listed above
└── tests/                      # data integrity, split leakage checks
```
