"""Create transparent PR and tile-bootstrap artifacts from an evaluator run.

The BEV evaluator reports micro-aggregated counts and writes the individual
NMS detections.  This companion deliberately operates on those retained
outputs: it neither reruns inference nor infers missing false negatives.

The bootstrap resamples *tiles*, not individual objects.  It is therefore a
descriptive sampling-sensitivity analysis.  It must not be presented as a
geographic confidence interval when the source tiles touch or otherwise share
acquisition context.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--source-label", default="", help="label used in the PR figure title")
    return parser.parse_args()


def metric_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, ...]:
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


def write_pr_curve(eval_dir: Path, gt_count: int, source_label: str) -> dict:
    """Write the exact ranked PR curve at the evaluator's rIoU=0.30 matching."""
    path = eval_dir / "detections.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["match_status"] != "ignored"]
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    scores = np.asarray([float(r["score"]) for r in rows], dtype=float)
    is_tp = np.asarray([r["match_status"] == "tp" for r in rows], dtype=bool)
    tp = np.cumsum(is_tp, dtype=int)
    fp = np.cumsum(~is_tp, dtype=int)
    recall = tp / max(gt_count, 1)
    precision = tp / np.maximum(tp + fp, 1)
    interpolated = np.maximum.accumulate(precision[::-1])[::-1]

    csv_path = eval_dir / "precision_recall_rIoU030.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rank", "score", "tp_cum", "fp_cum", "precision", "recall", "interpolated_precision"))
        for i in range(len(scores)):
            writer.writerow((i + 1, scores[i], tp[i], fp[i], precision[i], recall[i], interpolated[i]))

    # The all-point interpolated AP definition used by the evaluator.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    changed = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    ap = float(np.sum((mrec[changed] - mrec[changed - 1]) * mpre[changed]))

    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    ax.step(recall, precision, where="post", color="#1f77b4", linewidth=1.2, label="raw ranked curve")
    ax.step(recall, interpolated, where="post", color="#d62728", linewidth=1.4, label="all-point envelope")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title(f"{source_label} test, rotated IoU = 0.30, AP = {ap:.4f}")
    fig.savefig(eval_dir / "precision_recall_rIoU030.png", dpi=220)
    plt.close(fig)
    return {"ranked_detections": int(len(rows)), "gt_boxes": int(gt_count), "ap_rIoU_0.30": ap}


def main() -> None:
    args = parse_args()
    eval_dir = args.evaluation_dir.resolve()
    rows_path = eval_dir / "tile_metrics.csv"
    with rows_path.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if abs(float(r["conf"]) - args.confidence) < 1e-12]
    if not rows:
        raise ValueError(f"No tile_metrics rows at confidence {args.confidence}")
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    tp = np.asarray([int(r["tp"]) for r in rows], dtype=np.int64)
    fp = np.asarray([int(r["fp"]) for r in rows], dtype=np.int64)
    fn = np.asarray([int(r["fn"]) for r in rows], dtype=np.int64)
    gt = int(sum(int(r["gt_boxes"]) for r in rows))
    point_p, point_r, point_f1 = metric_counts(tp.sum(), fp.sum(), fn.sum())

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(0, len(rows), size=(args.replicates, len(rows)))
    sample_tp = tp[sample_indices].sum(axis=1)
    sample_fp = fp[sample_indices].sum(axis=1)
    sample_fn = fn[sample_indices].sum(axis=1)
    p, r, f1 = metric_counts(sample_tp, sample_fp, sample_fn)
    qs = (0.025, 0.5, 0.975)
    quantiles_by_metric = np.quantile(np.stack((p, r, f1)), qs, axis=1).T
    bootstrap = {
        name: {"lower_2.5": float(q[0]), "median": float(q[1]), "upper_97.5": float(q[2])}
        for name, q in zip(("precision", "recall", "f1"), quantiles_by_metric, strict=True)
    }
    if args.source_label:
        source_label = args.source_label
    else:
        summary_path = eval_dir / "summary.json"
        source_label = str(json.loads(summary_path.read_text(encoding="utf-8")).get("source", "dataset")) if summary_path.exists() else "dataset"
    pr = write_pr_curve(eval_dir, gt, source_label)
    report = {
        "purpose": "Tile-resampling uncertainty sensitivity and full ranked PR curve from retained evaluator output.",
        "evaluation_dir": str(eval_dir),
        "confidence": args.confidence,
        "match_iou": args.match_iou,
        "tiles": len(rows),
        "gt_boxes": gt,
        "point_estimate": {"tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()), "precision": float(point_p), "recall": float(point_r), "f1": float(point_f1)},
        "bootstrap": {"unit": "tile", "replicates": args.replicates, "seed": args.seed, "percentile_interval": "2.5th--97.5th", "metrics": bootstrap},
        "caveat": "This tile bootstrap assumes exchangeable tiles and summarises sampling sensitivity; it is not a geographic/generalisation confidence interval.",
        "pr_curve": pr,
    }
    (eval_dir / "bootstrap_tile_ci.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
