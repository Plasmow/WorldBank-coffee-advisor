"""Quality report for the AI chain: translation, then analyze() end to end.

Not a pytest file (pytest only collects test_*.py): it loads the real translation
model and talks to Ollama, and prints a report a human reads.

    python scripts/report_chain.py              # everything
    python scripts/report_chain.py translate    # Luganda -> English only
    python scripts/report_chain.py analyze      # full chain only
    python scripts/report_chain.py templates    # English -> Luganda on the SMS templates

The Luganda cases are written the way a farmer on Mount Elgon would text:
"emmwanyi" is the coffee plant ("kaawa" is the drink), short, no punctuation.
Each case carries a reference English translation and the expected label.
"""

import json
import pathlib
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai import translate  # noqa: E402
from ai.analyze import analyze  # noqa: E402

# (luganda, reference english, expected label, expected decision)
# expected decision: "answer" for clear disease/healthy descriptions,
# "escalate" or "clarify" for other/vague -- never "answer" on those.
LG_CASES = [
    # leaf_rust
    ("Ebikoola by'emmwanyi zange birina obuwunga obwa langi ya kacungwa wansi waabyo",
     "The leaves of my coffee have orange powder underneath them", "leaf_rust", "answer"),
    ("Wansi w'ebikoola waliwo enfuufu eya kyenvu",
     "Under the leaves there is yellow dust", "leaf_rust", "answer"),
    ("Ebikoola bigwa nnyo era biriko amabala aga kyenvu",
     "The leaves are falling a lot and have yellow spots", "leaf_rust", "answer"),
    ("Ebikoola birina amabala ga kyenvu waggulu ate wansi buwunga bwa kacungwa",
     "The leaves have yellow spots on top and orange powder underneath", "leaf_rust", "answer"),
    # phoma
    ("Ensonda z'ebikoola ebito zifuuse nzirugavu era zikala",
     "The tips of the young leaves have turned black and are drying", "phoma", "answer"),
    ("Ebikoola birina amabala amaddugavu ku mbiriizi, obudde bwa mpewo n'enkuba",
     "The leaves have black spots on the edges, the weather is cold and rainy", "phoma", "answer"),
    ("Amatabi amato gakala okuva waggulu nga gafuuka maddugavu",
     "The young shoots are drying from the top and turning black", "phoma", "answer"),
    # healthy
    ("Ebikoola by'emmwanyi zange bya kiragala era bimasamasa, tewali mabala",
     "The leaves of my coffee are green and shiny, there are no spots", "healthy", "answer"),
    ("Emmwanyi zange ziri bulungi, ebikoola biramu bulungi",
     "My coffee is fine, the leaves are healthy", "healthy", "answer"),
    # other
    ("Ebiwuka birya ebikoola by'emmwanyi",
     "Insects are eating the coffee leaves", "other", "escalate"),
    ("Emmwanyi zigula ssente mmeka leero?",
     "How much does coffee sell for today?", "other", "escalate"),
    ("Ebibala by'emmwanyi biriko obutuli",
     "The coffee berries have small holes", "other", "escalate"),
    ("Emiti gy'emmwanyi gyonna giwotoka",
     "All the coffee trees are wilting", "other", "escalate"),
    ("Oli otya ssebo",
     "How are you sir", "other", "escalate"),
    # vague: must not get advice
    ("Emmwanyi zange zirabika bubi",
     "My coffee looks bad", "other", "clarify"),
]

EN_CASES = [
    ("orange powder under the leaves", "leaf_rust", "answer"),
    ("yellow spots on top of the leaves and orange dust below, many leaves falling", "leaf_rust", "answer"),
    ("the tips of the young leaves are turning black and drying", "phoma", "answer"),
    ("dark brown patches at the leaf edges after the cold rain", "phoma", "answer"),
    ("my leaves are dark green and shiny, no spots", "healthy", "answer"),
    ("small holes in the berries", "other", "escalate"),
    ("what is the price of coffee today?", "other", "escalate"),
    ("my coffee looks strange", "other", "clarify"),
]

# Clarify flow: vague first message, then the answer to the precision question.
CLARIFY_CASES = [
    ("Emmwanyi zange zirabika bubi", "Ku bikoola, maddugavu ku nsonda", "phoma"),
    ("my coffee looks strange", "orange powder under the leaves", "leaf_rust"),
]

SAFE_FOR_OTHER = {"clarify", "escalate"}


# ---------- a small chrF, so no extra dependency ----------

def _ngrams(s, n):
    s = " ".join(s.lower().split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyp, ref, max_n=6, beta=2):
    """Character n-gram F-score (Popovic 2015), 0-100. Rough but language-blind."""
    precs, recs = [], []
    for n in range(1, max_n + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


# ---------- sections ----------

def run_translate():
    print("\n=== Luganda -> English (Opus-MT fine-tune, CTranslate2 int8) ===")
    t0 = time.perf_counter()
    translate.load()
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    scores, times = [], []
    for lg, ref, *_ in LG_CASES:
        t0 = time.perf_counter()
        hyp = translate.lug_to_en(lg)
        times.append(time.perf_counter() - t0)
        score = chrf(hyp, ref)
        scores.append(score)
        print(f"\n  lg : {lg}\n  ref: {ref}\n  mt : {hyp}\n  chrF {score:5.1f}   {times[-1]:.2f}s")

    print(f"\nmean chrF {sum(scores) / len(scores):.1f}   "
          f"mean latency {sum(times) / len(times):.2f}s/sentence")


def run_analyze():
    print("\n=== analyze(): full chain ===")
    rows = []
    for text, label, decision in [(lg, lab, dec) for lg, _, lab, dec in LG_CASES] + EN_CASES:
        t0 = time.perf_counter()
        r = analyze(text)
        dt = time.perf_counter() - t0
        rows.append((text, label, decision, r, dt))
        ok_label = r["label"] == label
        safe = r["decision"] == decision or (label == "other" and r["decision"] in SAFE_FOR_OTHER)
        flag = "OK " if ok_label and safe else ("SAFE" if r["decision"] != "answer" else "BAD")
        print(f"\n  [{flag}] ({r['lang']}) {text}\n"
              f"         text_en: {r['text_en']}\n"
              f"         expected {label}/{decision}  got clf={r['label']}@{r['proba']} "
              f"llm={r['llm_label']} -> {r['decision']} ({r['reason']}) {r['template_id']}  {dt:.1f}s")

    print("\n--- clarify flow ---")
    for first, answer, label in CLARIFY_CASES:
        r1 = analyze(first)
        r2 = analyze(first, answer)
        print(f"\n  {first!r} -> {r1['decision']}\n  + {answer!r}\n    text_en: {r2['text_en']}\n"
              f"    expected {label}  got clf={r2['label']}@{r2['proba']} llm={r2['llm_label']} "
              f"-> {r2['decision']} {r2['template_id']}")

    # Summary
    def summary(name, subset):
        n = len(subset)
        clf = sum(r["label"] == lab for _, lab, _, r, _ in subset)
        llm = sum(r["llm_label"] == lab for _, lab, _, r, _ in subset)
        answered = [(lab, r) for _, lab, _, r, _ in subset if r["decision"] == "answer"]
        wrong_advice = sum(r["label"] != lab for lab, r in answered)
        should_answer = sum(dec == "answer" for _, _, dec, _, _ in subset)
        got_answer = sum(dec == "answer" and r["decision"] == "answer" and r["label"] == lab
                         for _, lab, dec, r, _ in subset)
        lat = sum(dt for *_, dt in subset) / n
        print(f"  {name:8s} n={n:2d}  classifier acc {clf}/{n}  llm acc {llm}/{n}  "
              f"correct advice {got_answer}/{should_answer}  WRONG advice {wrong_advice}  "
              f"mean {lat:.1f}s")

    print("\n--- summary ---")
    summary("luganda", rows[:len(LG_CASES)])
    summary("english", rows[len(LG_CASES):])
    print("  decisions:", dict(Counter(r["decision"] for *_, r, _ in rows)))


def run_templates():
    """What Noor actually receives, for a Luganda speaker to check.

    No model here any more: the translator only goes lg -> en, and every
    message going out is written by hand in data/templates.json.
    """
    print("\n=== SMS templates, en + lg (for a human to check) ===")
    templates = json.loads((ROOT / "data" / "templates.json").read_text(encoding="utf-8"))
    unverified = 0
    for key, value in templates.items():
        if not isinstance(value, dict):
            continue
        mark = "" if value.get("lg_verified") else "   [lg NOT verified]"
        unverified += 0 if value.get("lg_verified") else 1
        print(f"\n  {key}{mark}\n    en: {value.get('en', '')}\n    lg: {value.get('lg', '')}")
    print(f"\n  {unverified} of {len(templates)} still need a native speaker's eye.")


SECTIONS = {"translate": run_translate, "analyze": run_analyze, "templates": run_templates}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SECTIONS)
    for name in wanted:
        SECTIONS[name]()
