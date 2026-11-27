"""Aggregate the archived vehicle detection counts by source or method."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def summarize(path: Path, group: str) -> dict[str, tuple[int, int, int]]:
    totals = defaultdict(lambda: [0, 0, 0])
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            tp, fp, fn = (int(row[key]) for key in ("tp", "fp", "fn"))
            if min(tp, fp, fn) < 0 or tp + fn != int(row["gt"]):
                raise ValueError(f"Inconsistent detection counts in {path.name}, tile {row['tile']}")
            total = totals[row[group]]
            total[0] += tp
            total[1] += fp
            total[2] += fn
    return {key: (tp, fp, fn) for key, (tp, fp, fn) in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", action="store_true", help="summarize DALES segmentation baselines")
    args = parser.parse_args()
    filename, group = (
        ("dales-segmentation-baselines.csv", "method")
        if args.baselines else ("seed-123-test-counts.csv", "source")
    )
    for label, (tp, fp, fn) in summarize(ROOT / "results" / filename, group).items():
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        print(f"{label:18} TP={tp:4} FP={fp:4} FN={fn:4} "
              f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    main()
