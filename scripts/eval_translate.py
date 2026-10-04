#!/usr/bin/env python
"""Measure the Luganda -> English translator on real human pairs.

    python scripts/eval_translate.py [-n 30] [--pairs data/synthetic/domain_pairs.jsonl]

Prints every line (Luganda, what the model said, what a human wrote), then
the numbers that belong in the datasheet: chrF, load time, median latency per
SMS, resident memory, model size on disk.

chrF needs sacrebleu, a development dependency on purpose -- the server never
scores anything:  pip install sacrebleu
"""
import argparse
import json
import pathlib
import resource
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_PAIRS = ROOT / "data" / "synthetic" / "domain_pairs.jsonl"


def rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return rss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def folder_mb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=30, help="how many pairs to translate")
    ap.add_argument("--pairs", type=pathlib.Path, default=DEFAULT_PAIRS)
    args = ap.parse_args()

    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = [p for p in pairs if p.get("lg") and p.get("en")][: args.n]
    if not pairs:
        sys.exit(f"no usable pairs in {args.pairs}")

    from ai import translate

    before = rss_mb()
    load_seconds = translate.load()

    outputs, latencies = [], []
    for i, pair in enumerate(pairs, 1):
        started = time.perf_counter()
        english = translate.lug_to_en(pair["lg"])
        latencies.append((time.perf_counter() - started) * 1000)
        outputs.append(english)
        print(f"\n{i:3}. LG  {pair['lg']}")
        print(f"     MT  {english}")
        print(f"     REF {pair['en']}")

    print("\n" + "=" * 72)
    try:
        from sacrebleu.metrics import CHRF

        chrf = CHRF().corpus_score(outputs, [[p["en"] for p in pairs]]).score
        print(f"chrF                 {chrf:.1f}")
    except ImportError:
        print("chrF                 (pip install sacrebleu to score)")

    print(f"pairs                {len(pairs)}")
    print(f"load                 {load_seconds:.1f} s")
    print(f"latency per SMS      {statistics.median(latencies):.0f} ms median, "
          f"{max(latencies):.0f} ms worst")
    print(f"resident memory      {rss_mb():.0f} MB  (+{rss_mb() - before:.0f} MB for the model)")
    print(f"model on disk        {folder_mb(translate.model_dir()):.0f} MB  "
          f"({translate.model_dir().relative_to(ROOT)})")


if __name__ == "__main__":
    main()
