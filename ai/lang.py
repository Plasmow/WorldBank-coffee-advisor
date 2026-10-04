"""Luganda or English? Pure Python, no model: app/ may import it freely.

Counts common English words against Luganda markers (function words and the
noun-class prefixes that start most Luganda words). Ties go to `default`,
so a farmer's known language survives an ambiguous "ok" or "zzzz".
"""
import re

EN_WORDS = {
    "the", "a", "an", "is", "are", "was", "my", "on", "of", "and", "with", "in",
    "to", "it", "they", "what", "how", "there", "some", "under", "have", "has",
    "i", "me", "you", "this", "that", "not", "no", "yes", "look", "looks", "see",
    "leaf", "leaves", "coffee", "spots", "spot", "yellow", "orange", "brown",
    "black", "powder", "tree", "trees", "plant", "plants", "green", "dry",
    "dying", "turning", "price", "much", "hello", "hi", "please", "help",
    "berries", "berry", "today", "bad", "good", "strange", "wrong", "something",
}
LG_WORDS = {
    "nga", "ku", "mu", "era", "naye", "oba", "kiki", "ki", "leero", "nnyo",
    "bulungi", "bubi", "zange", "byange", "wange", "kyange", "gyange", "zino",
    "bino", "emmwanyi", "mmwanyi", "kaawa", "kawa", "ebikoola", "ekikoola",
    "amabala", "ssente", "mmeka", "bbeeyi", "beeyi", "wansi", "waggulu",
    "oli", "otya", "gyebale", "ssebo", "nnyabo", "webale", "kale", "tewali",
    "langi", "kyenvu", "kacungwa", "nzirugavu", "maddugavu", "kiragala",
}
# Noun-class / verb prefixes; only counted on words long enough to carry one.
LG_PREFIX = re.compile(
    r"^(eki|ebi|emi|omu|aba|ama|oku|olu|obu|ag|eb|ez|ek|zi|bi|gi|ki|ly|by|zz|mm|nn|ss|tt|bb|kk|gg)"
)


def detect_lang(text: str, default: str = "en") -> str:
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    words = [w for w in (w.strip("'") for w in words) if w]
    if not words:
        return default
    en = sum(w in EN_WORDS for w in words)
    lg = sum(w in LG_WORDS or (len(w) >= 5 and w not in EN_WORDS and bool(LG_PREFIX.match(w)))
             for w in words)
    # Luganda elisions (by'emmwanyi, w'ebikoola) look nothing like English ones.
    lg += sum(bool(re.match(r"^[a-z]{1,3}'[aeiou]", w)) for w in words)
    if en == lg:
        return default
    return "en" if en > lg else "lg"
