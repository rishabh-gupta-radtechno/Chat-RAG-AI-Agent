"""
Language detection utilities.
Detects Hindi (Devanagari script) vs English based on Unicode character ranges.
"""

import re

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

LANG_ENGLISH = "en"
LANG_HINDI = "hi"

_LANG_NAMES = {
    LANG_ENGLISH: "English",
    LANG_HINDI: "Hindi",
}


def detect_language(text: str) -> str:
    """Return 'hi' if text contains Devanagari characters, else 'en'."""
    return LANG_HINDI if _DEVANAGARI_RE.search(text) else LANG_ENGLISH


def language_name(lang: str) -> str:
    """Return human-readable name for a language code."""
    return _LANG_NAMES.get(lang, "English")
