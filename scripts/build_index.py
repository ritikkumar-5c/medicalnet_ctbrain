"""Build a manifest CSV of series-level samples.

Layout expected:
    data/{normal,near_normal,abnormal}/{study_id}/{series_id}/*.dcm

Two ways to build the manifest:

  (A) Reuse the SAME splits as ct_brain_classifier (preferred — keeps the two
      models on identical patient partitions for a fair comparison):
        python scripts/build_index.py \
            --from-splits /root/ritikkumar/train_data/csvs/splits \
            --out data/manifest.csv
      Reads {train,val,test}.csv (columns: path,slice_size,label) produced by
      ct_brain_classifier/data/split_classifier_csv.py and copies their split
      assignment verbatim.

  (B) Scan a data root and make a fresh study-grouped, class-stratified split:
        python scripts/build_index.py --data-root /root/ritikkumar/train_data \
            --out data/manifest.csv

Every series folder with >= --min-slices DICOM files becomes one row.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

CLASS_TO_LABEL = {"normal": 0, "near_normal": 1, "abnormal": 2}


def rows_from_ctbrain_splits(splits_dir, min_slices):
    """Build manifest rows from ct_brain_classifier's split CSVs.

    Each split CSV has columns: path (absolute series folder), slice_size, label
    (class name). The split is taken from the file the row came from, so the
    train/val/test partition is identical to ct_brain_classifier's.
    """
    rows, skipped, missing = [], 0, 0
    for split in ("train", "val", "test"):
        csv_path = os.path.join(splits_dir, f"{split}.csv")
        if not os.path.isfile(csv_path):
            print(f"[warn] {csv_path} not found — skipping {split}")
            continue
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                series_dir = str(r["path"]).rstrip("/")
                label_name = str(r["label"]).strip().lower()
                if label_name not in CLASS_TO_LABEL:
                    raise SystemExit(f"unknown label {label_name!r} in {csv_path}")
                n = int(float(r.get("slice_size", 0) or 0))
                if n < min_slices:
                    skipped += 1
                    continue
                if not os.path.isdir(series_dir):
                    missing += 1
                    continue
                rows.append({
                    "series_dir": os.path.abspath(series_dir),
                    "study_id": os.path.basename(os.path.dirname(series_dir)),
                    "series_id": os.path.basename(series_dir),
                    "n_slices": n,
                    "label": CLASS_TO_LABEL[label_name],
                    "label_name": label_name,
                    "split": split,
                })
    if missing:
        print(f"[warn] {missing} series in the split CSVs no longer exist on disk — skipped")
    return rows, skipped


def scan(data_root, min_slices):
    rows = []
    skipped = 0
    for cls, label in CLASS_TO_LABEL.items():
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for study_dir in sorted(glob.glob(os.path.join(cls_dir, "*"))):
            if not os.path.isdir(study_dir):
                continue
            study_id = os.path.basename(study_dir)
            series_dirs = [d for d in sorted(glob.glob(os.path.join(study_dir, "*")))
                           if os.path.isdir(d)]
            if not series_dirs:  # dcm directly under study
                series_dirs = [study_dir]
            for sd in series_dirs:
                n = len(glob.glob(os.path.join(sd, "*.dcm")))
                if n < min_slices:
                    skipped += 1
                    continue
                rows.append({
                    "series_dir": os.path.abspath(sd),
                    "study_id": study_id,
                    "series_id": os.path.basename(sd),
                    "n_slices": n,
                    "label": label,
                    "label_name": cls,
                })
    return rows, skipped


def split_by_study(rows, val_frac, test_frac, seed):
    """Deterministic study-grouped, class-stratified split."""
    import random
    rng = random.Random(seed)

    # study_id -> label (a study has a single class)
    study_label = {}
    for r in rows:
        study_label[r["study_id"]] = r["label"]

    by_label = defaultdict(list)
    for study, lab in study_label.items():
        by_label[lab].append(study)

    assign = {}
    for lab, studies in by_label.items():
        studies = sorted(studies)
        rng.shuffle(studies)
        n = len(studies)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        for i, s in enumerate(studies):
            if i < n_test:
                assign[s] = "test"
            elif i < n_test + n_val:
                assign[s] = "val"
            else:
                assign[s] = "train"
    for r in rows:
        r["split"] = assign[r["study_id"]]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-splits", default=None,
                    help="dir with ct_brain_classifier {train,val,test}.csv; reuse their "
                         "exact split (preferred). Overrides --data-root scan/split.")
    ap.add_argument("--data-root", default="/root/ritikkumar/train_data",
                    help="(fallback) scan this root and make a fresh study-grouped split")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--min-slices", type=int, default=10,
                    help="skip series with fewer slices (scouts/localizers)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.from_splits:
        rows, skipped = rows_from_ctbrain_splits(args.from_splits, args.min_slices)
        if not rows:
            raise SystemExit(f"No usable rows from splits in {args.from_splits}.")
    else:
        rows, skipped = scan(args.data_root, args.min_slices)
        if not rows:
            raise SystemExit(f"No series found under {args.data_root}. Is data downloaded?")
        rows = split_by_study(rows, args.val_frac, args.test_frac, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = ["series_dir", "study_id", "series_id", "n_slices",
              "label", "label_name", "split"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Summary
    from collections import Counter
    print(f"Wrote {len(rows)} series to {args.out}  (skipped {skipped} thin series)")
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == split]
        c = Counter(r["label_name"] for r in sub)
        studies = len({r["study_id"] for r in sub})
        print(f"  {split:5s}: {len(sub):4d} series / {studies:4d} studies  "
              + " ".join(f"{k}={c.get(k,0)}" for k in CLASS_TO_LABEL))


if __name__ == "__main__":
    main()
