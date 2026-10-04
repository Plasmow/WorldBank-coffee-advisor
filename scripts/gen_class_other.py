#!/usr/bin/env python3
"""Generate the 'other' class: farmer-style SMS that are NOT leaf rust, NOT
phoma and NOT a healthy leaf. The photo dataset has no such class, so the
texts are generated from a list of topics (edit TOPICS below, or pass --topics).

Each topic is one GROUP for the train/calib/test split (scripts/make_splits.py):
all messages of a topic end up on the same side, so the test set contains
problems the classifier never saw during training.

Run from the repo root (same environment as describe_images.py):
    python scripts/gen_other.py --dry-run                    # plan + cost estimate, no API call
    python scripts/gen_other.py --only berry_darkening greetings_thanks --yes   # pilot
    python scripts/gen_other.py --max-cost 2 --yes           # full run, stops at $2

Output: data/synthetic/other_messages.jsonl (one line per batch of messages).
Re-running skips batches already done. All texts are SYNTHETIC: say so in
docs/datasheet.md. Claude is used here only to build training data.
"""
import argparse
import difflib
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from leafdesc_common import DEFAULT_MODEL, resolve_prices

DEFAULT_OUT = Path("data/synthetic/other_messages.jsonl")
TOOL_NAME = "record_messages"

# --------------------------------------------------------------------------
# Topics. kind = "other_problem" (agronomic, not rust/phoma) or
# "offtopic_or_service" (not a plant problem). Edit freely; ids must be unique.
# Vague messages ("my coffee is sick") are left out ON PURPOSE: they should end
# up in the clarification / "not sure" path, so test them in hard_test.csv.
# --------------------------------------------------------------------------
TOPICS = [
    # Other coffee problems (plausible on Mount Elgon; verify in your fiche for 'other')
    {"id": "berry_darkening", "kind": "other_problem",
     "brief": "green or ripening cherries with dark, sunken spots; cherries turning black and dropping early"},
    {"id": "stem_tunnelling", "kind": "other_problem",
     "brief": "wood dust at the base of the trunk, small holes or tunnels in the stem, a branch or the whole tree yellowing and dying"},
    {"id": "leaf_mining", "kind": "other_problem",
     "brief": "pale winding trails or tan blotches inside the leaf, as if something is eating between the leaf surfaces"},
    {"id": "brown_ringed_spots", "kind": "other_problem",
     "brief": "round brown spots on leaves with a pale centre and a yellow ring around them, sometimes on cherries too"},
    {"id": "pale_leaves", "kind": "other_problem",
     "brief": "leaves pale green or yellow all over, or yellow between the veins, older leaves first, slow growth"},
    {"id": "dry_spell_stress", "kind": "other_problem",
     "brief": "leaves drooping, curling and drying at the edges during a dry or very hot spell"},
    {"id": "storm_damage", "kind": "other_problem",
     "brief": "torn, scorched or hail-damaged leaves; burnt-looking leaf edges after strong sun, wind or hail"},
    {"id": "cherry_holes", "kind": "other_problem",
     "brief": "tiny holes in ripe cherries and damaged beans inside, cherries dropping, poor-quality beans"},
    {"id": "bugs_on_cherries", "kind": "other_problem",
     "brief": "insects or bugs sucking on cherries or young shoots, cherries misshapen, beans with a bad taste"},
    {"id": "tree_wilting", "kind": "other_problem",
     "brief": "a whole tree or a main branch wilting and dying quickly, leaves hanging, bark peeling at the base"},
    {"id": "few_cherries", "kind": "other_problem",
     "brief": "trees that look fine but have few flowers or few cherries, flowers dropping, low harvest"},
    {"id": "seedling_losses", "kind": "other_problem",
     "brief": "nursery seedlings falling over, wilting or dying at the base of the stem"},
    # Not a plant diagnosis: must go to the agent, never be answered automatically
    {"id": "price_questions", "kind": "offtopic_or_service",
     "brief": "asking about today's coffee price or what the buyer should pay, in free text, without any keyword"},
    {"id": "harvest_drying", "kind": "offtopic_or_service",
     "brief": "when to pick, how to dry or store parchment, mould on drying coffee, how to improve quality"},
    {"id": "other_crops", "kind": "offtopic_or_service",
     "brief": "problems with her bananas, maize or beans (yellowing maize, bean leaf spots, pests in stored beans)"},
    {"id": "fertilizer_pesticide_requests", "kind": "offtopic_or_service",
     "brief": "asking which chemical to buy, how much to spray or mix, where to find fertilizer or fungicide, asking for doses"},
    {"id": "planting_weather", "kind": "offtopic_or_service",
     "brief": "when to plant or prune, whether rain is coming, how far apart to plant, shade trees"},
    {"id": "greetings_thanks", "kind": "offtopic_or_service",
     "brief": "greetings, thanks, good morning, 'ok', 'yes I understand', blessings"},
    {"id": "help_requests", "kind": "offtopic_or_service",
     "brief": "asking for the extension officer to call or visit, how the service works, whether the advice is free"},
    {"id": "unrelated_noise", "kind": "offtopic_or_service",
     "brief": "wrong-number messages, random words, test messages, jokes, family news, school fees, mobile money"},
]

ANGLES = [
    "very short messages of 3 to 8 words",
    "single-sentence reports",
    "questions ending with a question mark",
    "messages that say how long ago it was noticed (days, weeks)",
    "messages that mention the part of the plant or the place on the farm",
    "messages with typos and no punctuation",
    "messages that sound worried or urgent",
    "messages that sound calm and curious",
]

SYSTEM_PROMPT = """You help build a training set for a text classifier that reads SMS from smallholder Arabica coffee farmers on Mount Elgon, Uganda. Messages reach the classifier as short English text (typed in English, or machine-translated from Luganda). The class you generate is "other": messages that are NOT about leaf rust, NOT about Phoma leaf spot and NOT about a healthy-looking leaf. They are always passed to a human extension agent and never answered automatically.

Rules:
1. Write realistic SMS as a farmer would send them, not as an expert: usually under 160 characters, some only 3 to 8 words, simple English, some with spelling mistakes, missing punctuation, lowercase, or slightly awkward machine-translated wording. Do not make every message polite.
2. Describe what the farmer sees or asks. Do not use technical disease or pest names, and never use the words "rust", "phoma" or "fungus", or "orange powder". For plant problems, use everyday words for the symptoms.
3. Vary the writer (young or old, man or woman), the plant part, the level of worry, telling versus asking, and the amount of detail. No two messages may start with the same words.
4. Do not include advice, answers, or the farmer's own diagnosis of the cause.
5. No personal data: no phone numbers or real names.
6. Return exactly the requested number of messages."""

BANNED_RE = re.compile(r"\b(?:rust\w*|phoma|hemileia)\b|orange (?:powder|dust)", re.I)


def tool_schema(n: int) -> dict:
    return {
        "name": TOOL_NAME,
        "description": "Record the generated SMS messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "messages": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n}
            },
            "required": ["messages"],
        },
    }


def build_request(topic: dict, angles: list[str], n: int) -> dict:
    angle_lines = "\n".join(f"- {a}" for a in angles)
    text = (f"Topic: {topic['brief']}\n"
            f"Write {n} different SMS messages on this topic, spread across these styles:\n{angle_lines}")
    return {
        "system": SYSTEM_PROMPT,
        "tools": [tool_schema(n)],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "messages": [{"role": "user", "content": text}],
    }


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def clean_messages(raw: list, ) -> tuple[list[str], dict]:
    """Strip, drop banned words, drop exact / near duplicates inside the batch."""
    kept: list[str] = []
    dropped = {"banned": 0, "duplicate": 0, "empty_or_long": 0}
    for m in raw:
        if not isinstance(m, str):
            continue
        m = re.sub(r"\s+", " ", m).strip().strip('"\u201c\u201d')
        if not m or len(m) > 320:
            dropped["empty_or_long"] += 1
        elif BANNED_RE.search(m):
            dropped["banned"] += 1
        elif any(_norm(m) == _norm(k) or difflib.SequenceMatcher(None, _norm(m), _norm(k)).ratio() > 0.9
                 for k in kept):
            dropped["duplicate"] += 1
        else:
            kept.append(m)
    return kept, dropped


def generate_batch(client, args, topic: dict, b_idx: int, n: int) -> dict:
    rng = random.Random(f"{args.seed}:{topic['id']}:{b_idx}")  # same batch -> same styles on re-run
    angles = rng.sample(ANGLES, 3)
    req = build_request(topic, angles, n)
    resp = client.messages.create(model=args.model, max_tokens=120 + 60 * n, **req)
    block = next((b for b in resp.content if b.type == "tool_use" and b.name == TOOL_NAME), None)
    if block is None:
        raise ValueError(f"no tool_use block (stop_reason={resp.stop_reason})")
    msgs, dropped = clean_messages(block.input.get("messages", []))
    if not msgs:
        raise ValueError("no usable message after cleaning")
    return {
        "batch_id": f"{topic['id']}-{b_idx}",
        "group": topic["id"],  # the unit of the train/calib/test split
        "kind": topic["kind"],
        "label": "other",
        "messages": msgs,
        "requested": n,
        "dropped": dropped,
        "angles": angles,
        "model": args.model,
        "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        "synthetic": True,
        "source": "LLM-generated from a topic description (no photo, no real farmer message)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", type=Path, default=None,
                    help="JSON file [{id, kind, brief}, ...] replacing the built-in TOPICS")
    ap.add_argument("--only", nargs="+", default=None, help="only these topic ids (pilot run)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-topic", type=int, default=30, help="messages requested per topic")
    ap.add_argument("--batch-size", type=int, default=15, help="messages per API call")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cost", type=float, default=None, help="stop once real spend reaches this many USD")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the estimate, then exit")
    args = ap.parse_args()

    topics = json.loads(args.topics.read_text()) if args.topics else TOPICS
    if len({t["id"] for t in topics}) != len(topics):
        raise SystemExit("Topic ids must be unique")
    if args.only:
        unknown = set(args.only) - {t["id"] for t in topics}
        if unknown:
            raise SystemExit(f"Unknown topic id(s): {sorted(unknown)}")
        topics = [t for t in topics if t["id"] in set(args.only)]

    done = set()
    if args.out.exists():
        done = {json.loads(l)["batch_id"] for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()}
    jobs = []  # (topic, batch index, messages requested)
    for t in topics:
        n_batches = -(-args.per_topic // args.batch_size)
        for b in range(n_batches):
            n = min(args.batch_size, args.per_topic - b * args.batch_size)
            if f"{t['id']}-{b}" not in done:
                jobs.append((t, b, n))
    if not jobs:
        print("Nothing left to generate.")
        return

    # Estimate (rough: ~3.5 characters per token; ~38 output tokens per message)
    sample = build_request(jobs[0][0], ANGLES[:3], args.batch_size)
    in_tok = (len(sample["system"]) + len(json.dumps(sample["tools"])) + len(sample["messages"][0]["content"])) / 3.5
    out_tok = sum(60 + 38 * n for _, _, n in jobs) / len(jobs)
    prices = resolve_prices(args.model, args.price_in, args.price_out)
    print(f"{len(topics)} topics, {len(jobs)} API calls to do ({len(done)} batches already done), "
          f"~{sum(n for _, _, n in jobs)} messages requested")
    print(f"Per call: ~{in_tok:.0f} input tokens, ~{out_tok:.0f} output tokens")
    if prices:
        cost = len(jobs) * (in_tok * prices[0] + out_tok * prices[1]) / 1e6
        print(f"Estimated cost: ~${cost:.3f} (prices ${prices[0]}/M in, ${prices[1]}/M out: verify them)")
    else:
        print("No price known for this model: pass --price-in and --price-out to see a cost.")
    if args.dry_run:
        return
    if args.max_cost is not None and not prices:
        raise SystemExit("--max-cost needs known prices: pass --price-in and --price-out")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set the ANTHROPIC_API_KEY environment variable first")
    if not args.yes and input("Proceed with the API calls? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    client = anthropic.Anthropic(max_retries=6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    err_path = args.out.with_suffix(".errors.jsonl")
    lock, stop = threading.Lock(), threading.Event()
    stats = {"ok": 0, "err": 0, "msgs": 0, "in": 0, "out": 0, "spent": 0.0}

    def worker(job):
        topic, b, n = job
        if stop.is_set():
            return
        try:
            rec = generate_batch(client, args, topic, b, n)
            line, path, ok = json.dumps(rec, ensure_ascii=False), args.out, True
        except Exception as e:  # keep going: failed batches are retried on the next run
            line = json.dumps({"batch_id": f"{topic['id']}-{b}", "error": f"{type(e).__name__}: {e}"},
                              ensure_ascii=False)
            path, ok, rec = err_path, False, None
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if ok:
                stats["ok"] += 1
                stats["msgs"] += len(rec["messages"])
                stats["in"] += rec["usage"]["input_tokens"]
                stats["out"] += rec["usage"]["output_tokens"]
                if prices:
                    stats["spent"] = (stats["in"] * prices[0] + stats["out"] * prices[1]) / 1e6
                    if args.max_cost is not None and stats["spent"] >= args.max_cost:
                        stop.set()
            else:
                stats["err"] += 1
            done_n = stats["ok"] + stats["err"]
            if done_n % 5 == 0 or done_n == len(jobs):
                print(f"[{done_n}/{len(jobs)}] ok={stats['ok']} errors={stats['err']} "
                      f"messages={stats['msgs']} spent=${stats['spent']:.3f}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(worker, j) for j in jobs]):
            fut.result()

    print(f"\nDone: {stats['ok']} batches, {stats['msgs']} messages kept, {stats['err']} errors "
          f"(see {err_path}), real spend ${stats['spent']:.3f}")
    if stop.is_set():
        print("Stopped early: --max-cost reached. Re-run to continue where it stopped.")


if __name__ == "__main__":
    main()
