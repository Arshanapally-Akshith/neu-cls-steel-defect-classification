"""Phase 5 entry point: Grad-CAM verification for the Phase 3 ResNet18 model.

Selects which test images to explain purely from Phase 4's per-sample
predictions table (reports/error_analysis_predictions.csv) — never
re-derives correctness or re-runs training. Neither the model nor the
frozen test split are modified (checksummed before/after, like Phase 3/4).

Usage: python -m scripts.run_phase5_gradcam
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.loader import load_split_images
from src.data.split import SPLIT_NAMES
from src.gradcam.gradcam import GradCAM, border_energy_fraction, overlay_heatmap, resize_cam
from src.gradcam.report import generate_gradcam_report_markdown
from src.gradcam.selection import build_gradcam_selection
from src.gradcam.visualize import plot_image_grid
from src.models.transfer import build_transforms, load_trained_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_split_files(split_dir: Path) -> dict:
    hashes = {name: _sha256(split_dir / f"{name}.csv") for name in SPLIT_NAMES}
    hashes["split_summary.json"] = _sha256(split_dir / "split_summary.json")
    return hashes


def _has_category(row: dict, category: str) -> bool:
    return category in row["category"].split(",")


def _summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"n": int(len(arr)), "mean": float(arr.mean()), "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())}


def main() -> None:
    config = load_config()
    classes = config["dataset"]["classes"]
    gc_cfg = config["gradcam"]
    tl_cfg = config["transfer_learning"]

    split_dir = resolve_path(config["split"]["output_dir"])
    split_hashes_before = _hash_split_files(split_dir)
    resnet_model_path = resolve_path(config["transfer_learning"]["output"]["model_path"])
    model_hash_before = _sha256(resnet_model_path)

    predictions_csv_path = resolve_path(gc_cfg["output"]["predictions_csv"])
    predictions_df = pd.read_csv(predictions_csv_path)
    print(f"[load] Phase 4 predictions table: {len(predictions_df)} test images from {predictions_csv_path}")

    selection_df = build_gradcam_selection(
        predictions_df, classes, gc_cfg["confusion_prone_classes"], gc_cfg["n_extra_correct_per_confusion_class"]
    )
    print(f"[select] {len(selection_df)} unique images selected for Grad-CAM")
    for cat in ("correct_representative", "confusion_prone_correct", "incorrect"):
        n = sum(cat in c.split(",") for c in selection_df["category"])
        print(f"  {cat}: {n}")

    raw_dir = resolve_path(config["dataset"]["raw_dir"])
    mini_manifest = pd.DataFrame({"filename": selection_df["filename"], "class": selection_df["true_class"]})
    raw_images, _labels = load_split_images(mini_manifest, raw_dir, grayscale=False)
    assert list(_labels) == list(selection_df["true_class"]), "image load order mismatch"

    model, ckpt_classes, _ckpt = load_trained_model(resnet_model_path)
    assert ckpt_classes == classes
    class_to_idx = {c: i for i, c in enumerate(classes)}
    _train_t, eval_t = build_transforms(tl_cfg)
    input_size = tuple(tl_cfg["input_size"])

    reports_dir = resolve_path(gc_cfg["output"]["results_dir"])
    heatmaps_dir = resolve_path(gc_cfg["output"]["heatmaps_dir"])
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with GradCAM(model, model.layer4[-1]) as gc:
        for pos in range(len(selection_df)):
            row = selection_df.iloc[pos].to_dict()
            pil_img = Image.fromarray(raw_images[pos])
            input_tensor = eval_t(pil_img).unsqueeze(0)
            display_img = np.array(pil_img.resize(input_size, Image.BILINEAR))

            pred_idx = class_to_idx[row["resnet_pred"]]
            cam_pred = gc.generate(input_tensor, class_idx=pred_idx)
            overlay_pred = overlay_heatmap(display_img, resize_cam(cam_pred["cam"], input_size), alpha=gc_cfg["overlay_alpha"])
            border_pred = border_energy_fraction(cam_pred["cam"])

            stem = Path(row["filename"]).stem
            pred_path = heatmaps_dir / f"{stem}_pred-{row['resnet_pred']}.png"
            Image.fromarray(overlay_pred).save(pred_path)

            record = {
                "filename": row["filename"], "true_class": row["true_class"], "resnet_pred": row["resnet_pred"],
                "resnet_confidence": float(row["resnet_confidence"]), "category": row["category"],
                "cam_pred_probability": cam_pred["probability"], "border_energy_fraction_pred": border_pred,
                "overlay_pred_path": str(pred_path.relative_to(reports_dir)).replace("\\", "/"),
            }

            is_incorrect = _has_category(row, "incorrect")
            if is_incorrect:
                true_idx = class_to_idx[row["true_class"]]
                cam_true = gc.generate(input_tensor, class_idx=true_idx)
                overlay_true = overlay_heatmap(display_img, resize_cam(cam_true["cam"], input_size), alpha=gc_cfg["overlay_alpha"])
                border_true = border_energy_fraction(cam_true["cam"])
                true_path = heatmaps_dir / f"{stem}_true-{row['true_class']}.png"
                Image.fromarray(overlay_true).save(true_path)
                record.update({
                    "cam_true_probability": cam_true["probability"],
                    "border_energy_fraction_true": border_true,
                    "overlay_true_path": str(true_path.relative_to(reports_dir)).replace("\\", "/"),
                })

            record["_overlay_pred_img"] = overlay_pred
            if is_incorrect:
                record["_overlay_true_img"] = overlay_true
            results.append(record)

    print(f"[gradcam] generated {sum('_overlay_true_img' in r for r in results) + len(results)} CAM(s) over {len(results)} images")

    correct_records = [r for r in results if _has_category(r, "correct_representative")]
    confusion_prone_records = [r for r in results if _has_category(r, "confusion_prone_correct")]
    incorrect_records = [r for r in results if _has_category(r, "incorrect")]

    # --- Summary grids -------------------------------------------------------------
    def _grid(records, path):
        images = [r["_overlay_pred_img"] for r in records]
        titles = [f"{r['true_class']}\nconf={r['resnet_confidence']:.2f} border={r['border_energy_fraction_pred']:.2f}" for r in records]
        plot_image_grid(images, titles, path, n_cols=3)

    correct_grid_path = heatmaps_dir / "correct_examples_grid.png"
    confusion_prone_grid_path = heatmaps_dir / "confusion_prone_grid.png"
    incorrect_grid_path = heatmaps_dir / "incorrect_examples_grid.png"

    _grid(correct_records, correct_grid_path)
    _grid(confusion_prone_records, confusion_prone_grid_path)

    incorrect_images, incorrect_titles = [], []
    for r in incorrect_records:
        incorrect_images.append(r["_overlay_pred_img"])
        incorrect_titles.append(f"{r['true_class']}->{r['resnet_pred']} (pred CAM)\nconf={r['resnet_confidence']:.2f} border={r['border_energy_fraction_pred']:.2f}")
        incorrect_images.append(r["_overlay_true_img"])
        incorrect_titles.append(f"{r['true_class']}->{r['resnet_pred']} (true CAM)\nborder={r['border_energy_fraction_true']:.2f}")
    plot_image_grid(incorrect_images, incorrect_titles, incorrect_grid_path, n_cols=4)

    print(f"[plot] grids saved: {correct_grid_path.name}, {confusion_prone_grid_path.name}, {incorrect_grid_path.name}")

    # --- Summary CSV -----------------------------------------------------------------
    summary_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = reports_dir / "gradcam_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[save] per-image Grad-CAM summary written to {summary_csv_path}")

    # --- Verify nothing touched ---------------------------------------------------
    split_hashes_after = _hash_split_files(split_dir)
    model_hash_after = _sha256(resnet_model_path)
    if split_hashes_before != split_hashes_after:
        raise RuntimeError("Frozen split manifests changed during Phase 5 — this must never happen.")
    if model_hash_before != model_hash_after:
        raise RuntimeError("Model file changed during Phase 5 — this must never happen.")
    print("[verify] frozen split manifests AND model file byte-identical before/after run (SHA-256 checked)")

    # --- Border-energy quantitative summary -----------------------------------------
    border_energy_summary = {
        "correct_representative (pred CAM)": _summarize([r["border_energy_fraction_pred"] for r in correct_records]),
        "confusion_prone_correct (pred CAM)": _summarize([r["border_energy_fraction_pred"] for r in confusion_prone_records]),
        "incorrect (pred CAM)": _summarize([r["border_energy_fraction_pred"] for r in incorrect_records]),
        "incorrect (true CAM)": _summarize([r["border_energy_fraction_true"] for r in incorrect_records]),
    }

    ctx = {
        "predictions_csv": str(predictions_csv_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "n_selected": len(results),
        "n_correct_representative": len(correct_records),
        "n_confusion_prone": len(confusion_prone_records),
        "n_incorrect": len(incorrect_records),
        "overlay_alpha": gc_cfg["overlay_alpha"],
        "correct_representative": [{k: v for k, v in r.items() if not k.startswith("_")} for r in correct_records],
        "confusion_prone_correct": [{k: v for k, v in r.items() if not k.startswith("_")} for r in confusion_prone_records],
        "incorrect": [{k: v for k, v in r.items() if not k.startswith("_")} for r in incorrect_records],
        "confusion_prone_classes": gc_cfg["confusion_prone_classes"],
        "border_energy_summary": border_energy_summary,
        "artifacts": {
            "correct_grid": "gradcam/correct_examples_grid.png",
            "confusion_prone_grid": "gradcam/confusion_prone_grid.png",
            "incorrect_grid": "gradcam/incorrect_examples_grid.png",
            "summary_csv": "gradcam_summary.csv",
            "heatmaps_dir": "gradcam",
        },
    }

    report_md = generate_gradcam_report_markdown(ctx)
    report_path = reports_dir / "gradcam_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[report] written to {report_path} (qualitative sections are placeholders pending visual inspection)")


if __name__ == "__main__":
    main()
