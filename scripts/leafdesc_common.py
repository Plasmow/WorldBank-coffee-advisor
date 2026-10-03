"""Shared helpers for describe_images.py and estimate_cost.py.

Pipeline: list images -> group near-duplicates (augmented copies) -> select ->
build the API request -> (estimate cost | call the API).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Defaults (adapt to your repo)
# --------------------------------------------------------------------------
DEFAULT_ROOT = Path("data/raw/dataset_without_coffee_wilt")
DEFAULT_GROUPS_CACHE = Path("data/processed/image_groups.json")
DEFAULT_OUT = Path("data/synthetic/leaf_descriptions.jsonl")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# USD per million tokens: (input, output).
# VERIFY on https://docs.claude.com (pricing page) before trusting any cost
# figure. For a model that is not listed, pass --price-in and --price-out.
PRICES = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
BATCH_DISCOUNT = 0.5  # Message Batches API is billed at about half price

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

# Folder-name keyword -> label (names match your labels.json)
LABEL_RULES = [("health", "healthy"), ("rust", "leaf_rust"), ("phoma", "phoma")]
# What Claude is told about the class (used only for a consistency check)
LABEL_HINTS = {"healthy": "healthy leaf", "leaf_rust": "coffee leaf rust", "phoma": "Phoma leaf spot"}

TOOL_NAME = "record_descriptions"

SYSTEM_PROMPT = """You help build a training set for a text classifier used by a farmer-advice service for smallholder Arabica coffee farmers on Mount Elgon, Uganda. Farmers will describe a coffee leaf by voice or SMS. Your job is to write realistic descriptions of leaf photos, in the words a farmer might use.

Rules:
1. Describe only what is visible in the photo. Never guess a cause, the weather, how long it has been there, how many leaves are affected, or what to do about it.
2. Never name a disease, fungus or pest, and never use the words "rust", "phoma", "fungus", "disease" or "infection". A farmer describing a leaf does not know the diagnosis. Colour words such as "orange" or "brown" are fine.
3. Use plain everyday English, the way a farmer without technical training would say it. Avoid technical terms such as lesion, necrotic, chlorosis, pustule, spore.
4. Follow the style hint given for each description. The descriptions must differ from each other in wording and structure, not only in length.
5. The dataset authors rotated, mirrored and brightness-adjusted the photos. Ignore orientation, and do not mention that a photo is dark, bright, flipped or upside down.
6. A class label from the dataset is provided only so you can check consistency. Never mention it. Set label_consistency to "yes" if the visible symptoms match it (for a healthy leaf: no marks), "partly" if they only weakly match, and "no" if they clearly do not. Set image_quality to "poor" if the photo is too blurry or dark to judge, or does not show a coffee leaf.
7. If nothing abnormal is visible, say so honestly instead of inventing symptoms."""

STYLES = [
    "very short, 6 to 12 words, like a text message",
    "one plain sentence in simple words",
    "two short sentences: first the colour of the leaf, then the marks on it",
    "say where on the leaf the marks are (edge, tip, middle, near the vein), only if marks are visible",
    "start with 'My coffee leaf' and describe it as a worried farmer would",
    "spoken style, as if said on a phone call, with a small hesitation such as 'it looks like'",
    "describe only colours and texture, with no guess about the cause",
    "describe the size and shape of the marks with everyday comparisons (a grain of maize, a coin), only if marks are visible",
]

# Words that should not appear in a farmer-style description (flagged, not dropped)
FLAG_RE = re.compile(
    r"\b(rust\w*|phoma|hemileia|fungus|fungal|disease\w*|infect\w*|upside|mirror\w*|flipped)\b", re.I
)

# --------------------------------------------------------------------------
# Listing + grouping
# --------------------------------------------------------------------------


@dataclass
class Item:
    id: str  # path relative to the dataset root
    path: Path
    label: str
    group: str = ""


def _label_from_path(path: Path, root: Path) -> str | None:
    for part in reversed(path.relative_to(root).parts[:-1]):  # nearest folder first
        low = part.lower()
        for key, label in LABEL_RULES:
            if key in low:
                return label
    return None


def list_images(root: Path) -> tuple[list[Item], int]:
    """All images under root with a recognised class folder.

    Files whose name contains ':' (the '*.jpg:Zone.Identifier' files that
    Windows downloads leave behind in WSL) are ignored.
    """
    items: list[Item] = []
    unlabeled = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS or ":" in p.name:
            continue
        label = _label_from_path(p, root)
        if label is None:
            unlabeled += 1
            continue
        items.append(Item(id=p.relative_to(root).as_posix(), path=p, label=label))
    return items, unlabeled


_HASH = 16  # 16x16 difference hash = 256 bits
_pop = int.bit_count if hasattr(int, "bit_count") else (lambda x: bin(x).count("1"))


def _dhash(img: Image.Image) -> int:
    g = ImageOps.autocontrast(img.convert("L"), cutoff=2)  # tolerant to brightness changes
    g = g.resize((_HASH + 1, _HASH), Image.LANCZOS)
    px = list(g.getdata())
    w = _HASH + 1
    v = 0
    for y in range(_HASH):
        for x in range(_HASH):
            v = (v << 1) | (px[y * w + x] > px[y * w + x + 1])
    return v


def _hashes_all_orientations(path: Path) -> list[int]:
    """Hash of the image under the 8 flips/rotations. Element 0 is the original."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        out = []
        for flip in (False, True):
            base = ImageOps.mirror(im) if flip else im
            for k in range(4):
                out.append(_dhash(base.rotate(90 * k, expand=True)))
    return out


def assign_groups(items: list[Item], threshold: int, cache_path: Path) -> None:
    """Cluster near-duplicates (flipped / rotated by 90 deg / brightness-changed
    copies of one photo) within each class. Sets item.group.

    Rotations by other angles (e.g. 15 deg) are NOT detected by this method.
    """
    ids = [it.id for it in items]
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache.get("threshold") == threshold and set(cache.get("groups", {})) == set(ids):
            for it in items:
                it.group = cache["groups"][it.id]
            return

    reps: dict[str, list[tuple[str, int]]] = defaultdict(list)  # label -> [(group, identity hash)]
    counters: Counter = Counter()
    for n, it in enumerate(items, 1):
        hs = _hashes_all_orientations(it.path)
        for gid, rep_hash in reps[it.label]:
            if min(_pop(rep_hash ^ h) for h in hs) <= threshold:
                it.group = gid
                break
        else:
            counters[it.label] += 1
            it.group = f"{it.label}-{counters[it.label]:04d}"
            reps[it.label].append((it.group, hs[0]))
        if n % 500 == 0:
            print(f"  hashed {n}/{len(items)} images...")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"threshold": threshold, "groups": {i.id: i.group for i in items}}, indent=1))


def group_report(items: list[Item]) -> None:
    groups: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        groups[it.group].append(it)
    print("Near-duplicate grouping (copies of the same photo count once):")
    for label in sorted({i.label for i in items}):
        n_img = sum(1 for i in items if i.label == label)
        sizes = [len(g) for gid, g in groups.items() if gid.startswith(label + "-")]
        print(f"  {label:10s} {n_img:5d} images -> {len(sizes):5d} groups "
              f"(largest group: {max(sizes)}, singletons: {sum(1 for s in sizes if s == 1)})")


def contact_sheet(items: list[Item], out_path: Path, top: int = 12, per_row: int = 8, thumb: int = 96) -> list[int]:
    """One row per group (largest groups first) so you can check by eye that
    grouped images really are copies of the same photo."""
    groups: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        groups[it.group].append(it)
    biggest = sorted(groups.values(), key=len, reverse=True)[:top]
    sheet = Image.new("RGB", (per_row * thumb, len(biggest) * thumb), "white")
    for r, g in enumerate(biggest):
        for c, it in enumerate(g[:per_row]):
            with Image.open(it.path) as im:
                sheet.paste(im.convert("RGB").resize((thumb, thumb)), (c * thumb, r * thumb))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return [len(g) for g in biggest]


def select_items(items: list[Item], per_class: int | None, one_per_group: bool, seed: int) -> list[Item]:
    """Optionally keep one random image per group, cap per class, and return a
    seeded shuffle (so --limit gives a mix of classes)."""
    rng = random.Random(seed)
    pool = items
    if one_per_group:
        groups: dict[str, list[Item]] = defaultdict(list)
        for it in items:
            groups[it.group].append(it)
        pool = [rng.choice(g) for _, g in sorted(groups.items())]
    by_label: dict[str, list[Item]] = defaultdict(list)
    for it in pool:
        by_label[it.label].append(it)
    out: list[Item] = []
    for label in sorted(by_label):
        lst = by_label[label]
        rng.shuffle(lst)
        out.extend(lst[:per_class] if per_class else lst)
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------
# Request building
# --------------------------------------------------------------------------


def sample_styles(rng: random.Random, n: int) -> list[str]:
    return rng.sample(STYLES, n)


def _tool_schema(n: int) -> dict:
    return {
        "name": TOOL_NAME,
        "description": "Record the farmer-style descriptions and the quality checks for this photo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_quality": {"type": "string", "enum": ["ok", "poor"]},
                "label_consistency": {"type": "string", "enum": ["yes", "partly", "no"]},
                "descriptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": n,
                    "maxItems": n,
                    "description": "One description per style hint, in the same order.",
                },
            },
            "required": ["image_quality", "label_consistency", "descriptions"],
        },
    }


def base_request(label: str, styles: list[str], image_b64: str, media_type: str) -> dict:
    """Arguments shared by messages.create and messages.count_tokens."""
    n = len(styles)
    style_lines = "\n".join(f"{i}. {s}" for i, s in enumerate(styles, 1))
    text = (
        f"Dataset class label: {LABEL_HINTS[label]}.\n"
        f"Write {n} descriptions of this leaf, one per style, in this order:\n{style_lines}"
    )
    return {
        "system": SYSTEM_PROMPT,
        "tools": [_tool_schema(n)],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                    {"type": "text", "text": text},
                ],
            }
        ],
    }


def encode_image(path: Path) -> tuple[str, str]:
    return base64.standard_b64encode(path.read_bytes()).decode(), MEDIA_TYPES[path.suffix.lower()]


# --------------------------------------------------------------------------
# Cost estimation
# --------------------------------------------------------------------------


def approx_image_tokens(w: int, h: int) -> int:
    """Documented rule of thumb: tokens ~ width*height/750 (long edge capped at 1568 px)."""
    longest = max(w, h)
    if longest > 1568:
        s = 1568 / longest
        w, h = w * s, h * s
    return min(math.ceil(w * h / 750), 1600)


def resolve_prices(model: str, price_in: float | None, price_out: float | None) -> tuple[float, float] | None:
    if price_in is not None and price_out is not None:
        return price_in, price_out
    return PRICES.get(model)


def default_max_tokens(n: int) -> int:
    return 200 + 120 * n


def _fixed_tokens_offline(n: int) -> float:
    """Rough token count of system prompt + tool schema + instruction text (~3.5 chars/token)."""
    req = base_request("healthy", STYLES[:n], "", "image/jpeg")
    req["messages"][0]["content"].pop(0)  # drop the image block
    return len(json.dumps(req)) / 3.5


def _fixed_tokens_exact(client, model: str, selected: list[Item], n: int, k: int = 3) -> float:
    """Ask the (free) count_tokens endpoint, subtract the image part, average over k images."""
    vals = []
    for it in selected[:k]:
        b64, media = encode_image(it.path)
        req = base_request(it.label, STYLES[:n], b64, media)
        total = client.messages.count_tokens(model=model, **req).input_tokens
        with Image.open(it.path) as im:
            vals.append(total - approx_image_tokens(*im.size))
    return statistics.mean(vals)


def read_log_usage(path: Path) -> tuple[float, float, int] | None:
    """Mean input/output tokens per call measured in a previous run's JSONL log."""
    if not path.exists():
        return None
    ins, outs = [], []
    for line in path.read_text().splitlines():
        if line.strip():
            u = json.loads(line).get("usage")
            if u:
                ins.append(u["input_tokens"])
                outs.append(u["output_tokens"])
    return (statistics.mean(ins), statistics.mean(outs), len(ins)) if ins else None


def estimate(selected: list[Item], n: int, max_tokens: int, prices, client=None, model: str = DEFAULT_MODEL,
             calibrate: Path | None = None) -> dict:
    """Return token and cost estimates for describing `selected`."""
    sizes = []
    for it in selected:
        with Image.open(it.path) as im:
            sizes.append(im.size)
    img_tokens = [approx_image_tokens(*s) for s in sizes]

    source = "offline estimate (chars/3.5 + image formula)"
    cal = read_log_usage(calibrate) if calibrate else None
    if cal:
        in_per_call, out_expected, n_cal = cal
        source = f"measured on {n_cal} real calls in {calibrate}"
    else:
        if client is not None:
            fixed = _fixed_tokens_exact(client, model, selected, n)
            source = "count_tokens (exact fixed part) + image formula"
        else:
            fixed = _fixed_tokens_offline(n)
        in_per_call = fixed + (statistics.mean(img_tokens) if img_tokens else 0)
        out_expected = 70 + 55 * n  # rough guess: ~55 tokens per description + JSON overhead

    calls = len(selected)
    res = {
        "calls": calls,
        "source": source,
        "in_per_call": in_per_call,
        "out_expected": out_expected,
        "out_max": max_tokens,
        "total_in": in_per_call * calls,
        "total_out_expected": out_expected * calls,
        "total_out_max": max_tokens * calls,
    }
    if prices:
        pin, pout = prices
        res["usd_expected"] = (res["total_in"] * pin + res["total_out_expected"] * pout) / 1e6
        res["usd_worst"] = (res["total_in"] * pin + res["total_out_max"] * pout) / 1e6
        res["usd_expected_batch"] = res["usd_expected"] * BATCH_DISCOUNT
    return res


def print_estimate(res: dict, model: str, prices, n: int) -> None:
    print("\n=== Cost estimate ===")
    print(f"Model: {model} | calls: {res['calls']} | descriptions per call: {n}"
          f" | descriptions total: {res['calls'] * n}")
    print(f"Input tokens/call : ~{res['in_per_call']:.0f}  ({res['source']})")
    print(f"Output tokens/call: ~{res['out_expected']:.0f} expected, {res['out_max']} max (max_tokens)")
    print(f"Totals: input ~{res['total_in']:,.0f} tokens, output ~{res['total_out_expected']:,.0f} expected "
          f"(<= {res['total_out_max']:,.0f})")
    if prices:
        print(f"Prices used: ${prices[0]}/M input, ${prices[1]}/M output (verify on the Anthropic pricing page)")
        print(f"Estimated cost: ${res['usd_expected']:.2f} expected | ${res['usd_worst']:.2f} worst case "
              f"| ~${res['usd_expected_batch']:.2f} expected with the Batch API (50% off)")
    else:
        print("No price known for this model: pass --price-in and --price-out (USD per million tokens).")
    print("====================\n")


# --------------------------------------------------------------------------
# Shared CLI
# --------------------------------------------------------------------------


def add_selection_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="dataset folder (searched recursively)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSONL of results; ids already in it are skipped")
    ap.add_argument("--groups-cache", type=Path, default=DEFAULT_GROUPS_CACHE)
    ap.add_argument("--dup-threshold", type=int, default=20,
                    help="max Hamming distance (of 256 bits) to call two images the same photo; lower = stricter")
    ap.add_argument("--all-images", action="store_true",
                    help="describe every image instead of one image per near-duplicate group")
    ap.add_argument("--per-class", type=int, default=None, help="cap on images per class (after grouping)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N of the shuffled selection (pilot run)")
    ap.add_argument("--n-variants", type=int, default=3, choices=range(1, len(STYLES) + 1), metavar="N",
                    help=f"descriptions per image, 1-{len(STYLES)} (each gets a different style)")
    ap.add_argument("--max-tokens", type=int, default=None, help="output cap per call (default 200 + 120*n)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    ap.add_argument("--seed", type=int, default=0)

def drop_unreadable(items: list[Item]) -> list[Item]:
    good, bad = [], []
    for it in items:
        try:
            with Image.open(it.path) as im:
                im.load()
            good.append(it)
        except Exception as e:
            bad.append((it, e))
    if bad:
        print(f"WARNING: {len(bad)} unreadable image(s) skipped:")
        for it, e in bad[:20]:
            print(f"  {it.id}: {type(e).__name__}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")
    return good

def prepare_selection(args) -> list[Item]:
    """List, group and select images; drop ids already present in --out."""
    if not args.root.exists():
        raise SystemExit(f"Dataset folder not found: {args.root}")
    items, unlabeled = list_images(args.root)
    if not items:
        raise SystemExit(f"No labelled images under {args.root} (check LABEL_RULES in leafdesc_common.py)")
    counts = Counter(i.label for i in items)
    print(f"Found {len(items)} images: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if unlabeled:
        print(f"  ({unlabeled} images ignored: no recognised class folder)")
    items = drop_unreadable(items)
    print("Grouping near-duplicates (cached after the first run)...")
    assign_groups(items, args.dup_threshold, args.groups_cache)
    group_report(items)

    selected = select_items(items, args.per_class, not args.all_images, args.seed)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text().splitlines() if l.strip()}
    selected = [it for it in selected if it.id not in done]
    if args.limit:
        selected = selected[: args.limit]
    print(f"Selected {len(selected)} images to describe ({len(done)} already done in {args.out})")
    return selected
