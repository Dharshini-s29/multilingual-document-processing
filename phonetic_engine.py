"""
phonetic_engine.py - Indic Phonetic Normalization Engine
Handles variations like vanakkam/wanakkam/vanakam -> canonical form.
"""
import re
from typing import Optional
from dataclasses import dataclass, field

try:
    import jellyfish
except ImportError:
    jellyfish = None

# Multi-char phoneme mappings (longest-first)
INDIC_PHONEME_MAP = [
    ("tth","T"),("thh","T"),("ddh","D"),("dh","D"),("bh","B"),("ph","F"),
    ("kh","K"),("gh","G"),("ch","C"),("chh","C"),("jh","J"),("th","T"),
    ("sh","S"),("shh","S"),("tt","T"),("dd","D"),("nn","N"),("ng","N"),
    ("zh","Z"),("w","V"),("aa","A"),("ee","I"),("ii","I"),("oo","U"),
    ("uu","U"),("ai","E"),("ei","E"),("au","O"),("ou","O"),("ey","E"),
    ("ay","E"),("ah","A"),("uh","A"),
]

INDIC_SINGLE_MAP = {
    "a":"A","e":"I","i":"I","o":"O","u":"U","b":"B","c":"K","d":"D",
    "f":"F","g":"G","h":"","j":"J","k":"K","l":"L","m":"M","n":"N",
    "p":"P","q":"K","r":"R","s":"S","t":"T","v":"V","w":"V","x":"KS",
    "y":"Y","z":"Z",
}

CANONICAL_DICTIONARY = {
    "vanakkam":"vanakkam","vanakam":"vanakkam","wanakkam":"vanakkam",
    "wanakam":"vanakkam","nandri":"nandri","nandree":"nandri",
    "nandhri":"nandri","eppadi":"eppadi","epdi":"eppadi","epadi":"eppadi",
    "irukenga":"irukkinga","irukinga":"irukkinga","irukkinga":"irukkinga",
    "romba":"romba","rumba":"romba",
    "namaste":"namaste","namasthe":"namaste","namastey":"namaste",
    "namashte":"namaste","dhanyavaad":"dhanyavaad","dhanyawad":"dhanyavaad",
    "dhaniyavad":"dhanyavaad","shukriya":"shukriya","sukriya":"shukriya",
    "accha":"accha","acha":"accha","achha":"accha",
    "bahut":"bahut","bohot":"bahut","bohut":"bahut",
    "theek":"theek","thik":"theek","teek":"theek",
    "kaise":"kaise","kese":"kaise",
    "namaskaram":"namaskaram","namaskaaramu":"namaskaram",
    "namaskara":"namaskara","namskara":"namaskara",
    "karenge":"karenge","karinge":"karenge",
    "bolna":"bolna","bolana":"bolna",
    "batana":"batana","bathana":"batana","batao":"batao","bathao":"batao",
}


@dataclass
class PhoneticMatch:
    original: str
    normalized: str
    phonetic_code: str
    confidence: float
    method: str


@dataclass
class PhoneticEngine:
    custom_dictionary: dict = field(default_factory=dict)
    similarity_threshold: float = 0.75

    def __post_init__(self):
        self._dictionary = {**CANONICAL_DICTIONARY, **self.custom_dictionary}
        self._phonetic_index: dict[str, list[str]] = {}
        for word in self._dictionary:
            code = self.indic_soundex(word)
            self._phonetic_index.setdefault(code, []).append(word)

    @staticmethod
    def indic_soundex(word: str, max_length: int = 6) -> str:
        cleaned = re.sub(r"[^a-z]", "", word.lower().strip())
        if not cleaned:
            return ""
        phonemes = []
        i = 0
        while i < len(cleaned):
            matched = False
            for pattern, code in INDIC_PHONEME_MAP:
                if cleaned[i:].startswith(pattern):
                    if code:
                        phonemes.append(code)
                    i += len(pattern)
                    matched = True
                    break
            if not matched:
                mapped = INDIC_SINGLE_MAP.get(cleaned[i], cleaned[i].upper())
                if mapped:
                    phonemes.append(mapped)
                i += 1
        if not phonemes:
            return cleaned[0].upper()
        collapsed = [phonemes[0]]
        for p in phonemes[1:]:
            if p != collapsed[-1]:
                collapsed.append(p)
        code = cleaned[0].upper() + "".join(collapsed)
        return code[:max_length].ljust(max_length, "0")

    @staticmethod
    def indic_metaphone(word: str) -> str:
        cleaned = re.sub(r"[^a-z]", "", word.lower().strip())
        if not cleaned:
            return ""
        result = []
        i = 0
        while i < len(cleaned):
            matched = False
            for pattern, code in INDIC_PHONEME_MAP:
                if cleaned[i:].startswith(pattern):
                    if code:
                        result.append(code)
                    i += len(pattern)
                    matched = True
                    break
            if not matched:
                mapped = INDIC_SINGLE_MAP.get(cleaned[i], "")
                if mapped:
                    result.append(mapped)
                i += 1
        return "".join(result)

    def phonetic_similarity(self, word1: str, word2: str) -> float:
        code1, code2 = self.indic_soundex(word1), self.indic_soundex(word2)
        if code1 == code2:
            return 1.0
        if jellyfish:
            return jellyfish.jaro_winkler_similarity(code1, code2)
        if not code1 or not code2:
            return 0.0
        common = sum(1 for a, b in zip(code1, code2) if a == b)
        return common / max(len(code1), len(code2))

    def normalize_word(self, word: str) -> PhoneticMatch:
        lower = word.lower().strip()
        if lower in self._dictionary:
            return PhoneticMatch(word, self._dictionary[lower],
                                self.indic_soundex(lower), 1.0, "dictionary")
        code = self.indic_soundex(lower)
        candidates = self._phonetic_index.get(code, [])
        if candidates:
            best, best_score = None, 0.0
            for c in candidates:
                s = self.phonetic_similarity(lower, c)
                if s > best_score:
                    best_score, best = s, c
            if best and best_score >= self.similarity_threshold:
                return PhoneticMatch(word, self._dictionary.get(best, best),
                                    code, round(best_score, 3), "soundex")
        meta = self.indic_metaphone(lower)
        best_match, best_sim = None, 0.0
        for dw, cf in self._dictionary.items():
            dm = self.indic_metaphone(dw)
            if jellyfish:
                sim = jellyfish.jaro_winkler_similarity(meta, dm)
            else:
                common = sum(1 for a, b in zip(meta, dm) if a == b)
                sim = common / max(len(meta), len(dm), 1)
            if sim > best_sim:
                best_sim, best_match = sim, cf
        if best_match and best_sim >= self.similarity_threshold:
            return PhoneticMatch(word, best_match, code,
                                round(best_sim, 3), "metaphone")
        return PhoneticMatch(word, word, code, 0.0, "none")

    def normalize_text(self, text: str) -> dict:
        tokens = re.findall(r"[\w]+|[^\w\s]|\s+", text)
        normalized_tokens, normalizations = [], []
        normalized_count, total_words = 0, 0
        for token in tokens:
            if re.match(r"^\w+$", token):
                total_words += 1
                match = self.normalize_word(token)
                normalized_tokens.append(match.normalized if match.confidence > 0 else token)
                if match.confidence > 0:
                    normalized_count += 1
                normalizations.append(vars(match))
            else:
                normalized_tokens.append(token)
        return {
            "original_text": text,
            "normalized_text": "".join(normalized_tokens),
            "normalizations": normalizations,
            "stats": {"total_words": total_words, "normalized_count": normalized_count},
        }

    def find_similar_words(self, word: str, vocabulary: list[str], top_k: int = 5) -> list[dict]:
        scores = [{"word": c, "similarity": round(self.phonetic_similarity(word, c), 3)} for c in vocabulary]
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]


_default_engine: Optional[PhoneticEngine] = None

def get_engine(**kwargs) -> PhoneticEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = PhoneticEngine(**kwargs)
    return _default_engine

def normalize(text: str) -> dict:
    return get_engine().normalize_text(text)
