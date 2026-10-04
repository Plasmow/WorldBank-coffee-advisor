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
            _cache[name] = json.loads(path.read_text() or "{}")
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


def prices_text(lang="en"):
    p = _load("prices.json")
    rows = p.get("prices") or []
    if not rows:
        return t("prices_unavailable", lang)
    body = ", ".join(f"{r['grade']} {r['ugx_per_kg']}" for r in rows)
    return t("prices", lang).format(date=p.get("date", ""), body=body)


def reload():
    _cache.clear()


def template_path():
    return os.fspath(ROOT / "data" / "templates.json")
