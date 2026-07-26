"""
code_mixed.py - Code-Mixed Text Processing Engine
Handles Tanglish (Tamil+English), Hinglish (Hindi+English), etc.
Provides script detection, token-level language ID, and code-mixing metrics.
"""
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Script(str, Enum):
    LATIN = "Latin"
    DEVANAGARI = "Devanagari"
    TAMIL = "Tamil"
    TELUGU = "Telugu"
    KANNADA = "Kannada"
    MALAYALAM = "Malayalam"
    BENGALI = "Bengali"
    GUJARATI = "Gujarati"
    GURMUKHI = "Gurmukhi"
    ODIA = "Odia"
    NUMERIC = "Numeric"
    PUNCTUATION = "Punctuation"
    UNKNOWN = "Unknown"


# Unicode block ranges for Indian scripts
SCRIPT_RANGES = {
    Script.DEVANAGARI: (0x0900, 0x097F),
    Script.TAMIL:      (0x0B80, 0x0BFF),
    Script.TELUGU:     (0x0C00, 0x0C7F),
    Script.KANNADA:    (0x0C80, 0x0CFF),
    Script.MALAYALAM:  (0x0D00, 0x0D7F),
    Script.BENGALI:    (0x0980, 0x09FF),
    Script.GUJARATI:   (0x0A80, 0x0AFF),
    Script.GURMUKHI:   (0x0A00, 0x0A7F),
    Script.ODIA:       (0x0B00, 0x0B7F),
}

# Common English stopwords (for quick lang-id heuristic)
ENGLISH_STOPWORDS = {
    "the","is","are","was","were","am","been","being","have","has","had",
    "do","does","did","will","would","shall","should","may","might","must",
    "can","could","a","an","and","but","or","not","no","so","if","then",
    "than","that","this","these","those","it","its","i","me","my","we",
    "our","you","your","he","him","his","she","her","they","them","their",
    "what","which","who","whom","where","when","how","why","all","each",
    "every","both","few","more","most","other","some","such","very","just",
    "also","too","only","own","same","of","in","to","for","with","on","at",
    "by","from","as","into","about","between","through","after","before",
    "above","below","up","down","out","off","over","under","again","here",
}

# Common Hindi words in Latin script (transliterated)
HINDI_MARKERS = {
    "hai","hain","ka","ki","ke","ko","se","me","mein","par","pe","bhi",
    "aur","ya","lekin","magar","agar","toh","to","nahi","na","mat","kya",
    "kaun","kab","kahan","kaise","kitna","kitne","kitni","yeh","ye","woh",
    "wo","ek","do","teen","char","paanch","acha","accha","theek","thik",
    "bahut","bohot","kuch","sab","log","wala","wali","wale","kar","karo",
    "karna","karta","karti","karte","raha","rahi","rahe","gaya","gayi",
    "gaye","hoga","hogi","honge","tha","thi","the",
}

# Common Tamil words in Latin script
TAMIL_MARKERS = {
    "naan","nee","avan","aval","enna","yenna","epdi","eppadi","inga",
    "anga","romba","rumba","nalla","nallaa","vanakkam","nandri","illa",
    "illai","iruku","irukku","theriyum","therla","panna","pannu","vaa",
    "poo","po","sollu","solra","paaru","paru","enna","enna","mattum",
    "thaan","dhan","la","le","ku","oda","kuda","um","pola","maari",
    "aana","aanaa","enakku","unakku","avanga","ivanga","vera","venum",
}


@dataclass
class TokenAnalysis:
    token: str
    script: Script
    language: str  # "en", "hi", "ta", "te", "unknown"
    confidence: float
    is_switch_point: bool = False


def detect_char_script(char: str) -> Script:
    """Detect script of a single character."""
    if char.isdigit():
        return Script.NUMERIC
    if unicodedata.category(char).startswith("P") or char in ".,;:!?-()[]{}\"'":
        return Script.PUNCTUATION

    cp = ord(char)
    for script, (low, high) in SCRIPT_RANGES.items():
        if low <= cp <= high:
            return script

    if char.isascii() and char.isalpha():
        return Script.LATIN
    return Script.UNKNOWN


def detect_token_script(token: str) -> Script:
    """Detect dominant script of a token."""
    script_counts: dict[Script, int] = {}
    for ch in token:
        s = detect_char_script(ch)
        if s not in (Script.NUMERIC, Script.PUNCTUATION, Script.UNKNOWN):
            script_counts[s] = script_counts.get(s, 0) + 1
    if not script_counts:
        if all(c.isdigit() for c in token if not c.isspace()):
            return Script.NUMERIC
        return Script.UNKNOWN
    return max(script_counts, key=script_counts.get)


def detect_token_language(token: str, script: Script) -> tuple[str, float]:
    """
    Detect language of a single token.
    Returns (language_code, confidence).
    """
    lower = token.lower().strip()

    # Native script → direct mapping
    native_map = {
        Script.DEVANAGARI: "hi", Script.TAMIL: "ta", Script.TELUGU: "te",
        Script.KANNADA: "kn", Script.MALAYALAM: "ml", Script.BENGALI: "bn",
        Script.GUJARATI: "gu", Script.GURMUKHI: "pa", Script.ODIA: "or",
    }
    if script in native_map:
        return native_map[script], 0.95

    # Latin script → heuristic language identification
    if script == Script.LATIN:
        if lower in ENGLISH_STOPWORDS:
            return "en", 0.9
        if lower in HINDI_MARKERS:
            return "hi", 0.85
        if lower in TAMIL_MARKERS:
            return "ta", 0.85
        # Heuristic: if the word looks "English-ish"
        if re.match(r"^[a-z]+$", lower) and len(lower) <= 3:
            return "en", 0.5
        return "en", 0.4  # default guess for Latin script

    return "unknown", 0.0


def analyze_code_mixing(text: str) -> dict:
    """
    Full code-mixing analysis of a text string.
    Returns token-level analysis, language distribution, and mixing metrics.
    """
    raw_tokens = re.findall(r"[\w]+", text)
    if not raw_tokens:
        return {
            "tokens": [], "languages_found": [],
            "language_distribution": {}, "code_mixing_index": 0.0,
            "switch_points": 0, "dominant_language": "unknown",
        }

    analyses: list[dict] = []
    lang_counts: dict[str, int] = {}
    prev_lang: Optional[str] = None
    switch_points = 0

    for token in raw_tokens:
        script = detect_token_script(token)
        lang, conf = detect_token_language(token, script)
        is_switch = prev_lang is not None and lang != prev_lang and lang != "unknown" and prev_lang != "unknown"
        if is_switch:
            switch_points += 1
        analyses.append({
            "token": token, "script": script.value,
            "language": lang, "confidence": round(conf, 2),
            "is_switch_point": is_switch,
        })
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        if lang != "unknown":
            prev_lang = lang

    total = sum(lang_counts.values())
    distribution = {k: round(v / total, 3) for k, v in lang_counts.items()}
    dominant = max(lang_counts, key=lang_counts.get)

    # Code-Mixing Index (CMI): 1 - (max_lang_count / total) for multilingual text
    max_count = max(lang_counts.values())
    cmi = round(1 - (max_count / total), 3) if total > 1 else 0.0

    return {
        "tokens": analyses,
        "languages_found": list(lang_counts.keys()),
        "language_distribution": distribution,
        "code_mixing_index": cmi,
        "switch_points": switch_points,
        "dominant_language": dominant,
    }


def detect_languages_in_text(text: str) -> list[dict]:
    """
    Detect all languages present in a text.
    Uses both script-based detection and token-level heuristics.
    """
    analysis = analyze_code_mixing(text)
    results = []
    for lang, proportion in analysis["language_distribution"].items():
        lang_names = {
            "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
            "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali",
            "gu": "Gujarati", "pa": "Punjabi", "or": "Odia",
        }
        results.append({
            "code": lang,
            "name": lang_names.get(lang, lang),
            "proportion": proportion,
        })
    results.sort(key=lambda x: x["proportion"], reverse=True)
    return results
