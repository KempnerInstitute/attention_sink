"""
Aggregate probe results across backbone seeds for ADE20K and Pascal VOC.

Variants without a _bb<N> suffix are treated as _bb9000.
For each base variant, computes mean and std of best_val_iou across the 3 runs.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np

DATASETS = {
    "ade20k": Path("ade20k_probes/results.jsonl"),
    "pascal_voc": Path("pascal_voc_2012_probes/results.jsonl"),
}

BB_PATTERN = re.compile(r"^(.+?)(?:_bb(\d+))?$")


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def extract_base_variant(variant: str) -> str:
    """Strip the _bb<N> suffix to get the base variant name."""
    m = BB_PATTERN.match(variant)
    return m.group(1)


def aggregate(records: list[dict]) -> list[dict]:
    """Group by base variant, compute mean and std of best_val_iou."""
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        base = extract_base_variant(r["variant"])
        groups[base].append(r["best_val_iou"])

    aggregated = []
    for base_variant, values in groups.items():
        arr = np.array(values)
        aggregated.append({
            "variant": base_variant,
            "mean_iou": float(arr.mean()),
            "std_iou": float(arr.std()),
            "n_runs": len(values),
        })
    return aggregated

def main():
    all_output = []
    for dataset_name, path in DATASETS.items():
        records = load_results(path)
        agg = aggregate(records)
        for entry in agg:
            entry["dataset"] = dataset_name
            all_output.append(entry)

    out_path = Path(__file__).parent / "results_aggregated_probes.jsonl"
    with open(out_path, "w") as f:
        for entry in all_output:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(all_output)} aggregated results to {out_path}")
    for entry in all_output:
        print(f"  {entry['dataset']:>12s} | {entry['variant']:<20s} | "
              f"IoU={entry['mean_iou']:.4f} ± {entry['std_iou']:.4f} (n={entry['n_runs']})")


if __name__ == "__main__":
    main()
