"""Evaluate the non-neural shadow-hole/blob detector on frozen DALES OBBs.

This is a deliberately training-free baseline.  It takes only the raw PLY
point cloud, creates geometric candidates (ground holes + watershed blobs),
fits their L-shape OBBs and evaluates them with the same one-to-one rotated-IoU
protocol as the learned detector.  Candidate scores are *not calibrated
probabilities*; therefore this baseline reports a single, score-free operating
point after deterministic rotated NMS rather than AP.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Permit the documented ``python scripts/bev/...`` invocation from a checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bordereye.ml.bev.core.inference import match_detections_one_to_one, nms
from bordereye.ml.bev.core.nn_contract import box_corners
from bordereye.ml.bev.evaluation import evaluate_dets, local_box_verts
from bordereye.vehicle_pipeline import CandidateConfig, detect_candidates, load_tile_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bev-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--tiles", default="", help="comma-separated DALES tile ids; empty = all test")
    return parser.parse_args()


def candidate_detections(tile, candidates: list) -> list[dict]:
    """Convert L-shape candidates into evaluator detections.

    A score of one is intentional: the hand-designed procedure supplies no
    learned confidence.  A deterministic tie breaker in the tiny epsilon term
    keeps NMS reproducible without claiming confidence calibration.
    """
    out = []
    for index, candidate in enumerate(candidates):
        ls = candidate.ls
        theta = float(np.deg2rad(ls.rotation_deg))
        out.append({
            "cx": float(ls.center[0]), "cy": float(ls.center[1]),
            "L": float(ls.length), "W": float(ls.width), "th": theta,
            "z": 0.0, "height": 0.0,
            "score": 1.0 - index * 1e-12,
            "source": candidate.source,
            "verts": box_corners(float(ls.center[0]), float(ls.center[1]),
                                 float(ls.length), float(ls.width), theta),
        })
    return out


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = {x.strip() for x in args.tiles.split(",") if x.strip()}
    npz_paths = sorted((args.bev_root / "test").glob("dales_*.npz"))
    if wanted:
        npz_paths = [p for p in npz_paths if p.stem.removeprefix("dales_") in wanted]
    if not npz_paths:
        raise FileNotFoundError("No selected DALES test NPZ tiles")

    rows = []
    cfg = CandidateConfig()
    for npz_path in npz_paths:
        tile_id = npz_path.stem.removeprefix("dales_")
        ply_path = args.raw_root / "test" / f"{tile_id}_new.ply"
        with np.load(npz_path, allow_pickle=False) as data:
            gt_boxes = np.asarray(data["boxes_valid"], dtype=np.float32).reshape(-1, 7)
            ignore_boxes = np.asarray(data["boxes_ignore"], dtype=np.float32).reshape(-1, 7)
        tile = load_tile_ply(ply_path, "test")
        raw = candidate_detections(tile, detect_candidates(tile, cfg))
        dets = nms(raw, args.nms_iou)
        gt_verts = local_box_verts(gt_boxes, np.zeros(2))
        ignore_verts = local_box_verts(ignore_boxes, np.zeros(2))
        _, best_det, best_ignore = match_detections_one_to_one(
            dets, gt_verts, ignore_verts, args.match_iou
        )
        metrics, _, _, _ = evaluate_dets(
            dets, gt_verts, ignore_verts, conf=0.0, match_iou=args.match_iou,
            precomputed_matches=(best_det, best_ignore),
        )
        rows.append({"tile_id": tile_id, "raw_candidates": len(raw), "nms_candidates": len(dets),
                     "gt_boxes": len(gt_boxes), **metrics})
        print(f"{tile_id}: candidates={len(raw)} nms={len(dets)} "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}")

    total = {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "detections")}
    total["precision"] = total["tp"] / max(total["tp"] + total["fp"], 1)
    total["recall"] = total["tp"] / max(total["tp"] + total["fn"], 1)
    total["f1"] = 2 * total["precision"] * total["recall"] / max(total["precision"] + total["recall"], 1e-12)
    report = {
        "method": "classical ground-shadow-hole + watershed-blob + L-shape OBB",
        "input": "raw DALESObjects PLY; no learned model and no BEV neural channels",
        "score_note": "unranked geometric candidates; AP is not applicable",
        "nms_iou": args.nms_iou, "match_iou": args.match_iou,
        "tiles": len(rows), "summary": total, "per_tile": rows,
    }
    (args.out / "classical_hole_baseline.json").write_text(json.dumps(report, indent=2) + "\n",
                                                               encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
