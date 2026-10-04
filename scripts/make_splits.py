#!/usr/bin/env python3
"""Flatten the synthetic texts and split them BY GROUP into train / calib / test.

Inputs  (data/synthetic/):
    leaf_descriptions.jsonl   from describe_images.py  (group = near-duplicate photo group)
    other_messages.jsonl      from gen_other.py        (group = topic)
Outputs (data/synthetic/):
    dataset.jsonl   one row per text: text, label, group, split, source_id, kind
    splits.json     group -> split. FROZEN: groups already in it never move, so
                    adding data later does not reshuffle your test set.

Why by group: copies of one photo (and messages of one topic) look alike. If they
sit on both sides of the split, the test score is inflated. Each class is split
separately (stratified), so every class has train, calib and test data.

Photos are split by FILE NUMBER, not by hash group: the augmented copies of one
original photo have close numbers (coffee dataset/phoma/2300_276.jpg, 2300_288.jpg,
...), and the near-duplicate hash misses copies that were zoomed / cropped / rotated.
So the numbers of each class are cut into blocks of --block consecutive numbers, each
block goes to one split, and images within --gap numbers of a block that belongs to
another split are dropped. Two images in different splits are then at least
--gap numbers apart. Other messages are still split by topic group.

    train : fit the classifier
    calib : calibrate the probabilities and choose the "not sure" threshold
    test  : final numbers only. Never tune anything on it.

Run from the repo root:  python scripts/make_splits.py
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from leafdesc_common import FLAG_RE

SYN = Path("data/synthetic")
SPLIT_NAMES = ("train", "calib", "test")
PRIORITY = ("test", "calib", "train")  # when a text is duplicated across splits, keep it in the first of these


NUM_RE = re.compile(r"_(\d+)\.[A-Za-z0-9]+$")


def image_number(source_id: str) -> int | None:
    """2300_276.jpg -> 276 (the part after the class offset), None if there is no number."""
    m = NUM_RE.search(source_id)
    return int(m.group(1)) if m else None


def split_key(r: dict, block: int) -> str:
    """Unit that is moved as a whole: a block of consecutive photo numbers, else the group."""
    n = image_number(r["source_id"]) if r["kind"] == "photo" else None
    return r["group"] if n is None else f"{r['label']}-blk{n // block:04d}"


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (missing: {path})")
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_rows(args, drops: Counter) -> list[dict]:
    rows = []
    for rec in read_jsonl(args.photo):
        if "error" in rec:
            continue
        if not args.keep_inconsistent and rec.get("label_consistency") == "no":
            drops["photo: label_consistency=no"] += len(rec["descriptions"])
            continue
        if not args.keep_inconsistent and rec.get("image_quality") == "poor":
            drops["photo: image_quality=poor"] += len(rec["descriptions"])
            continue
        for text in rec["descriptions"]:
            if not args.keep_flagged and FLAG_RE.search(text):
                drops["photo: flagged word (disease name, 'rust', ...)"] += 1
                continue
            rows.append({"text": text, "label": rec["label"], "group": rec["group"],
                         "source_id": rec["id"], "kind": "photo"})
    for rec in read_jsonl(args.other):
        if "error" in rec:
            continue
        for i, text in enumerate(rec["messages"]):
            rows.append({"text": text, "label": rec["label"], "group": rec["group"],
                         "source_id": f"{rec['batch_id']}#{i}", "kind": rec.get("kind", "other")})
    return rows


def hard_test_texts(path: Path) -> set[str]:
    """Texts of the hand-written test set, so none of them can leak into training."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        col = next((c for c in (reader.fieldnames or []) if c.lower() in ("text", "message", "sms")),
                   (reader.fieldnames or [None])[0])
        return {norm(r[col]) for r in reader if r.get(col)}


def assign(groups_by_label: dict, existing: dict, ratios: dict, seed: int) -> dict:
    """Keep existing assignments; place new groups, per class, so the row counts
    follow the ratios. New groups are taken in a seeded hash order (deterministic)."""
    assignment = dict(existing)
    for label, groups in sorted(groups_by_label.items()):
        counts: Counter = Counter()
        for g, w in groups.items():
            if g in assignment:
                counts[assignment[g]] += w
        placed = sum(counts.values())
        todo = sorted((g for g in groups if g not in assignment),
                      key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        for g in todo:
            w = groups[g]
            best = max(SPLIT_NAMES, key=lambda s: ratios[s] * (placed + w) - counts[s])
            assignment[g] = best
            counts[best] += w
            placed += w
    return assignment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", type=Path, default=SYN / "leaf_descriptions.jsonl")
    ap.add_argument("--other", type=Path, default=SYN / "other_messages.jsonl")
    ap.add_argument("--hard-test", type=Path, default=Path("data/eval/hard_test.csv"))
    ap.add_argument("--splits", type=Path, default=SYN / "splits.json")
    ap.add_argument("--out", type=Path, default=SYN / "dataset.jsonl")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "CALIB", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=150,
                    help="photos: numbers n//BLOCK form one block that goes to a single split")
    ap.add_argument("--gap", type=int, default=30,
                    help="photos: drop images within GAP numbers of a neighbouring block of another split")
    ap.add_argument("--keep-flagged", action="store_true", help="keep photo texts containing 'rust', disease names, ...")
    ap.add_argument("--keep-inconsistent", action="store_true",
                    help="keep photo texts whose label_consistency is 'no' or image_quality is 'poor'")
    ap.add_argument("--reset", action="store_true", help="IGNORE the frozen splits.json and reassign everything")
    args = ap.parse_args()

    total = sum(args.ratios)
    ratios = dict(zip(SPLIT_NAMES, (r / total for r in args.ratios)))
    drops: Counter = Counter()

    print("Loading...")
    rows = load_rows(args, drops)
    if not rows:
        raise SystemExit("No texts found. Run describe_images.py and gen_other.py first.")

    hard = hard_test_texts(args.hard_test)
    if hard:
        before = len(rows)
        rows = [r for r in rows if norm(r["text"]) not in hard]
        drops["identical to a text of hard_test.csv"] += before - len(rows)

    # A text that appears under two different labels is ambiguous: drop every copy.
    labels_of = defaultdict(set)
    for r in rows:
        labels_of[norm(r["text"])].add(r["label"])
    before = len(rows)
    rows = [r for r in rows if len(labels_of[norm(r["text"])]) == 1]
    drops["same text under two labels (ambiguous)"] += before - len(rows)

    # Group -> label sanity check, then split.
    group_label: dict[str, str] = {}
    for r in rows:
        if group_label.setdefault(r["group"], r["label"]) != r["label"]:
            raise SystemExit(f"Group {r['group']} appears under two labels: fix the group ids")
    weights: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        r["unit"] = split_key(r, args.block)
        weights[r["label"]][r["unit"]] += 1

    existing = {}
    if args.splits.exists() and not args.reset:
        existing = json.loads(args.splits.read_text())["assignment"]
        print(f"Reusing {len(existing)} frozen group assignments from {args.splits}")
    elif args.reset:
        print("--reset: ignoring any frozen assignment (any model evaluated before is no longer comparable)")
    assignment = assign(weights, existing, ratios, args.seed)
    for r in rows:
        r["split"] = assignment[r["unit"]]

    # Photos: drop images too close (in number) to a block of another split.
    if args.gap > 0:
        before = len(rows)
        kept = []
        for r in rows:
            n = image_number(r["source_id"]) if r["kind"] == "photo" else None
            if n is not None:
                b, pos = divmod(n, args.block)
                for nb, near in ((b - 1, pos < args.gap), (b + 1, pos >= args.block - args.gap)):
                    other = assignment.get(f"{r['label']}-blk{nb:04d}")
                    if near and other is not None and other != r["split"]:
                        break
                else:
                    kept.append(r)
                continue
            kept.append(r)
        rows = kept
        drops[f"photo within {args.gap} numbers of a block of another split"] += before - len(rows)

    # Duplicate texts across / inside splits: keep one copy, in the most protected split.
    before = len(rows)
    seen, kept = set(), []
    for r in sorted(rows, key=lambda r: PRIORITY.index(r["split"])):
        key = norm(r["text"])
        if key not in seen:
            seen.add(key)
            kept.append(r)
    rows = kept
    drops["duplicate text (kept once)"] += before - len(rows)

    # Safety check: no topic group on two sides. Photo hash groups may straddle if
    # several images of one group were described (--all): report, the numbers rule.
    sides = defaultdict(set)
    for r in rows:
        sides[r["group"]].add(r["split"])
    straddling = {g for g, sp in sides.items() if len(sp) > 1}
    assert not any(r["kind"] != "photo" and r["group"] in straddling for r in rows), "a group ended up in two splits"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r.pop("unit", None)
    rows.sort(key=lambda r: (r["split"], r["label"], r["group"], r["source_id"], r["text"]))
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    args.splits.write_text(json.dumps(
        {"seed": args.seed, "ratios": ratios, "assignment": dict(sorted(assignment.items()))}, indent=1))

    # Report
    print(f"\nWrote {len(rows)} texts to {args.out} and the group assignment to {args.splits}\n")
    labels = sorted({r["label"] for r in rows})
    print(f"{'class':12s}" + "".join(f"{s:>16s}" for s in SPLIT_NAMES) + "   (texts / groups)")
    warnings = []
    for lab in labels:
        cells = []
        for s in SPLIT_NAMES:
            sel = [r for r in rows if r["label"] == lab and r["split"] == s]
            cells.append(f"{len(sel):>7d} /{len({r['group'] for r in sel}):>4d}")
            if len(sel) < 30 and s != "train":
                warnings.append(f"{lab}/{s} has only {len(sel)} texts: its numbers will be noisy")
            if not sel:
                warnings.append(f"{lab} has NO text in {s}")
        print(f"{lab:12s}" + "".join(f"{c:>16s}" for c in cells))
    train_counts = Counter(r["label"] for r in rows if r["split"] == "train")
    if train_counts and max(train_counts.values()) > 2 * min(train_counts.values()):
        warnings.append(f"training classes are unbalanced {dict(train_counts)}: use class weights or cap the big ones")
    if drops:
        print("\nDropped texts:")
        for reason, n in drops.most_common():
            if n:
                print(f"  {n:5d}  {reason}")
    if straddling:
        warnings.append(f"{len(straddling)} photo hash groups have images in several splits "
                        "(far apart in number): check them with the contact sheet")
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print("  -", w)
    print("\nTrain on split=train, calibrate and choose the threshold on split=calib, report on split=test (once).")


if __name__ == "__main__":
    main()
