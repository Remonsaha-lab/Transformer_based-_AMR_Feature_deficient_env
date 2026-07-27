#!/usr/bin/env python
"""Utilities for Stage 2 active-learning scan selection.

The selection order is always worst-to-best: lower semantic mIoU comes first,
so the first added scans are the hardest semantic segmentation scans.
"""

import argparse
import csv
import math
import random
from pathlib import Path


def _read_ids(path):
    with Path(path).open("r") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _write_ids(path, ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for scan_id in ids:
            f.write(f"{scan_id}\n")


def _dedupe(ids):
    out = []
    seen = set()
    for scan_id in ids:
        if scan_id in seen:
            continue
        out.append(scan_id)
        seen.add(scan_id)
    return out


def _count_from_fraction(total, fraction):
    if fraction <= 0:
        raise ValueError("fraction must be > 0")
    if fraction <= 1:
        return max(1, int(math.ceil(total * fraction)))
    return min(total, int(fraction))


def make_initial(args):
    ids = _dedupe(_read_ids(args.all_ids))
    rng = random.Random(args.seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n = _count_from_fraction(len(shuffled), args.fraction)
    selected = shuffled[:n]
    _write_ids(args.output, selected)
    print(f"Wrote {len(selected)}/{len(ids)} initial scans to {args.output}")


def _read_scores(path, scan_col, semantic_col):
    rows = []
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        if scan_col not in reader.fieldnames:
            raise ValueError(f"Missing scan column '{scan_col}' in {path}")
        if semantic_col not in reader.fieldnames:
            raise ValueError(f"Missing semantic column '{semantic_col}' in {path}")
        for row in reader:
            scan_id = row[scan_col].strip()
            if not scan_id:
                continue
            rows.append((scan_id, float(row[semantic_col]), row))
    rows.sort(key=lambda x: (x[1], x[0]))
    return rows


def make_next(args):
    current = _dedupe(_read_ids(args.current_ids))
    current_set = set(current)
    ranked = _read_scores(args.scores, args.scan_col, args.semantic_col)
    score_by_scan = {scan_id: score for scan_id, score, _ in ranked}
    remaining_worst_first = [
        scan_id for scan_id, _, _ in ranked if scan_id not in current_set
    ]
    remaining_worst_first = _dedupe(remaining_worst_first)

    if args.add_count is not None:
        n_add = args.add_count
    else:
        all_count = len(current) + len(remaining_worst_first)
        target_count = _count_from_fraction(all_count, args.target_fraction)
        n_add = max(0, target_count - len(current))

    added = remaining_worst_first[:n_add]
    selected = _dedupe(current + added)
    selected.sort(key=lambda scan_id: (score_by_scan.get(scan_id, float("inf")), scan_id))
    _write_ids(args.output, selected)

    if args.added_output is not None:
        _write_ids(args.added_output, added)

    print(
        f"Wrote {len(selected)} scans to {args.output}; "
        f"added {len(added)} worst-semantic scans"
    )
    if added:
        print("Worst added scans:")
        for rank, scan_id in enumerate(added[: args.preview], start=1):
            print(f"{rank:04d} {scan_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Stage 2 active-learning scan id lists.")
    subparsers = parser.add_subparsers(required=True)

    init_parser = subparsers.add_parser(
        "init", help="Create the initial random subset, e.g. 10%.")
    init_parser.add_argument("--all-ids", required=True)
    init_parser.add_argument("--fraction", type=float, default=0.10)
    init_parser.add_argument("--seed", type=int, default=0)
    init_parser.add_argument("--output", required=True)
    init_parser.set_defaults(func=make_initial)

    next_parser = subparsers.add_parser(
        "next", help="Add lowest-semantic-mIoU scans from a Stage 2 CSV.")
    next_parser.add_argument("--current-ids", required=True)
    next_parser.add_argument("--scores", required=True)
    next_parser.add_argument("--target-fraction", type=float, default=0.20)
    next_parser.add_argument("--add-count", type=int)
    next_parser.add_argument("--scan-col", default="scan_id")
    next_parser.add_argument("--semantic-col", default="semantic_miou")
    next_parser.add_argument("--output", required=True)
    next_parser.add_argument("--added-output")
    next_parser.add_argument("--preview", type=int, default=20)
    next_parser.set_defaults(func=make_next)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
