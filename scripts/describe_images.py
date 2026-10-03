#!/usr/bin/env python3
"""Generate short farmer-style descriptions of coffee-leaf photos with Claude.

Setup:
    pip install anthropic pillow
    export ANTHROPIC_API_KEY=...        # never commit the key

Typical workflow:
    python scripts/estimate_cost.py --per-class 300          # 1. estimate (free)
    python scripts/describe_images.py --per-class 300 --limit 20 --yes   # 2. pilot
    python scripts/estimate_cost.py --per-class 300 --calibrate          # 3. refine
    python scripts/describe_images.py --per-class 300 --max-cost 5       # 4. full run

Results are appended to data/synthetic/leaf_descriptions.jsonl (one line per
image). Re-running skips images already done, so it is safe to interrupt.
All output is SYNTHETIC text: label it as such in docs/datasheet.md.
"""
import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from leafdesc_common import (
    FLAG_RE,
    TOOL_NAME,
    add_selection_args,
    base_request,
    default_max_tokens,
    encode_image,
    estimate,
    prepare_selection,
    print_estimate,
    resolve_prices,
    sample_styles,
)


def describe_one(client, args, it, max_tokens: int) -> dict:
    rng = random.Random(f"{args.seed}:{it.id}")  # same image -> same styles on re-run
    styles = sample_styles(rng, args.n_variants)
    b64, media = encode_image(it.path)
    req = base_request(it.label, styles, b64, media)
    resp = client.messages.create(model=args.model, max_tokens=max_tokens, **req)
    block = next((b for b in resp.content if b.type == "tool_use" and b.name == TOOL_NAME), None)
    if block is None:
        raise ValueError(f"no tool_use block (stop_reason={resp.stop_reason})")
    data = block.input
    descs = [d.strip() for d in data.get("descriptions", []) if isinstance(d, str) and d.strip()]
    if len(descs) != args.n_variants:
        raise ValueError(f"expected {args.n_variants} descriptions, got {len(descs)} (stop_reason={resp.stop_reason})")
    return {
        "id": it.id,
        "label": it.label,
        "group": it.group,  # use it to split train/test by group, never by image
        "descriptions": descs,
        "styles": styles,
        "image_quality": data.get("image_quality"),
        "label_consistency": data.get("label_consistency"),
        "flags": sorted({m.group(0).lower() for d in descs for m in FLAG_RE.finditer(d)}),
        "model": args.model,
        "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        "synthetic": True,
        "source": "LLM-generated from a photo of the Mendeley coffee leaf dataset (k36wnd6knb)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_args(ap)
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (lower it if you hit rate limits)")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="stop launching new calls once the real spend reaches this many USD")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--dry-run", action="store_true", help="print the estimate and exit (same as estimate_cost.py)")
    args = ap.parse_args()

    selected = prepare_selection(args)
    if not selected:
        print("Nothing left to describe.")
        return

    prices = resolve_prices(args.model, args.price_in, args.price_out)
    max_tokens = args.max_tokens or default_max_tokens(args.n_variants)
    print_estimate(estimate(selected, args.n_variants, max_tokens, prices, model=args.model),
                   args.model, prices, args.n_variants)
    if args.dry_run:
        return
    if args.max_cost is not None and not prices:
        raise SystemExit("--max-cost needs known prices: pass --price-in and --price-out for this model")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set the ANTHROPIC_API_KEY environment variable first")
    if not args.yes and input("Proceed with the API calls? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    client = anthropic.Anthropic(max_retries=6)  # the SDK retries 429/5xx with backoff
    args.out.parent.mkdir(parents=True, exist_ok=True)
    err_path = args.out.with_suffix(".errors.jsonl")
    lock, stop = threading.Lock(), threading.Event()
    stats = {"ok": 0, "err": 0, "in": 0, "out": 0, "spent": 0.0}

    def worker(it):
        if stop.is_set():
            return
        try:
            rec = describe_one(client, args, it, max_tokens)
            line = json.dumps(rec, ensure_ascii=False)
            path, ok = args.out, True
        except Exception as e:  # keep going: failed ids are retried on the next run
            line = json.dumps({"id": it.id, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
            path, ok, rec = err_path, False, None
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if ok:
                stats["ok"] += 1
                stats["in"] += rec["usage"]["input_tokens"]
                stats["out"] += rec["usage"]["output_tokens"]
                if prices:
                    stats["spent"] = (stats["in"] * prices[0] + stats["out"] * prices[1]) / 1e6
                    if args.max_cost is not None and stats["spent"] >= args.max_cost:
                        stop.set()
            else:
                stats["err"] += 1
            done = stats["ok"] + stats["err"]
            if done % 25 == 0 or done == len(selected):
                print(f"[{done}/{len(selected)}] ok={stats['ok']} errors={stats['err']} "
                      f"spent=${stats['spent']:.3f}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(worker, it) for it in selected]):
            fut.result()

    print(f"\nDone: {stats['ok']} images described, {stats['err']} errors "
          f"(see {err_path}), real spend ${stats['spent']:.3f}")
    if stop.is_set():
        print("Stopped early: --max-cost reached. Re-run to continue where it stopped.")


if __name__ == "__main__":
    main()
