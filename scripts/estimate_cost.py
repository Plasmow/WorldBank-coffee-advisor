#!/usr/bin/env python3
"""Estimate the cost of describe_images.py BEFORE running it.

Three levels of accuracy:
  1. python scripts/estimate_cost.py
        Offline, no API key, no spend. Rough (+/- 20%).
  2. python scripts/estimate_cost.py --exact
        Uses the count_tokens endpoint (free, needs ANTHROPIC_API_KEY) for the
        fixed part of the request. Input tokens become accurate.
  3. python scripts/describe_images.py --limit 20 --yes   (a few cents)
     python scripts/estimate_cost.py --calibrate
        Reads real token usage from that pilot's log: the most accurate,
        especially for output tokens.

Use the same selection options (--per-class, --n-variants, --all-images, ...)
as for describe_images.py so both look at the same images.
"""
import argparse
import os
from pathlib import Path

from leafdesc_common import (
    add_selection_args,
    contact_sheet,
    default_max_tokens,
    estimate,
    list_images,
    assign_groups,
    prepare_selection,
    print_estimate,
    resolve_prices,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_args(ap)
    ap.add_argument("--exact", action="store_true", help="use the free count_tokens endpoint (needs an API key)")
    ap.add_argument("--calibrate", action="store_true", help="use real token usage recorded in --out")
    ap.add_argument("--contact-sheet", type=int, metavar="N", default=0,
                    help="write a picture of the N largest near-duplicate groups to check the grouping by eye")
    args = ap.parse_args()

    selected = prepare_selection(args)

    if args.contact_sheet:
        items, _ = list_images(args.root)
        assign_groups(items, args.dup_threshold, args.groups_cache)
        out = args.groups_cache.parent / "dup_groups_preview.jpg"
        sizes = contact_sheet(items, out, top=args.contact_sheet)
        print(f"Wrote {out} (one row per group, largest first; group sizes: {sizes})")

    if not selected:
        print("Nothing left to describe.")
        return

    client = None
    if args.exact:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("--exact needs the ANTHROPIC_API_KEY environment variable")
        import anthropic

        client = anthropic.Anthropic()

    prices = resolve_prices(args.model, args.price_in, args.price_out)
    max_tokens = args.max_tokens or default_max_tokens(args.n_variants)
    res = estimate(selected, args.n_variants, max_tokens, prices, client=client, model=args.model,
                   calibrate=args.out if args.calibrate else None)
    print_estimate(res, args.model, prices, args.n_variants)


if __name__ == "__main__":
    main()
