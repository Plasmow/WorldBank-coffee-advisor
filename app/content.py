"""Loaders for the two data files. Every word sent to Noor comes from here."""

import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
_cache = {}


def _load(name):
    if name not in _cache:
        path = ROOT / "data" / name
        try:
            _cache[name] = json.loads(path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            _cache[name] = {}
    return _cache[name]


def t(key, lang="en"):
    """A template. Accepts both a flat string and a {en, lg} mapping, so P3 can
    add Luganda later without restructuring the file or breaking the backend."""
    v = _load("templates.json").get(key)
    if isinstance(v, dict):
        return v.get(lang) or v.get("en") or ""
    return v or ""


def _ugx(n):
    return f"{int(n):,}"


def prices_text(lang="en"):
    """prices.json as it is, in a fixed wrapper. No AI on this path."""
    p = _load("prices.json")
    rows = p.get("prices") or []
    if not rows:
        return t("prices_unavailable", lang)
    parts = []
    # Noor sells arabica parchment: the dearest grade first.
    for r in sorted(rows, key=lambda r: -r.get("ugx_per_kg", 0)):
        lo, hi = r.get("ugx_per_kg_min"), r.get("ugx_per_kg_max")
        amount = f"{_ugx(lo)}-{_ugx(hi)}" if lo and hi else _ugx(r["ugx_per_kg"])
        parts.append(f"{r['grade']} {amount}")
    return t("prices", lang).format(date=p.get("date", ""), body="; ".join(parts))


def english_of(text):
    """The exact English of a message we sent, or None if we did not send it.

    Every outgoing message comes from templates.json (or the price wrapper),
    so its English is known: no translation service needed, and no mistakes.
    """
    for lang in ("lg", "en"):
        if text == prices_text(lang):
            return prices_text("en")
    for v in _load("templates.json").values():
        if isinstance(v, dict) and text in (v.get("lg"), v.get("en")):
            return v.get("en")
    return None


def reload():
    _cache.clear()


def template_path():
    return os.fspath(ROOT / "data" / "templates.json")
