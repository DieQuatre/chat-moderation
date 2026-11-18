"""Rule baseline (area B): bad-word lexicon + minimum edit distance.

A message is offensive if any of its normalized words
  - is in the bad-word lexicon, or
  - is an UNKNOWN word (not in the known-word list) at edit distance <= 1 from a
    lexicon word. This is the classic spelling-correction rule: only non-words are
    corrected, so normal words like "think" are never matched to a bad word.
Bad-word lexicon: LDNOOBW "List of Dirty, Naughty, Obscene, and Otherwise Bad Words" (en), CC-BY 4.0.
Known-word list: words seen 3+ times in CLEAN training comments (models/known_words.txt).

Run:  python -m src.models.lexicon_rule      (builds the known-word list, scores the val split)
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.text.normalize import tokens

ROOT = Path(__file__).resolve().parents[2]
LEXICON_FILE = ROOT / "resources" / "bad_words_en.txt"
KNOWN_FILE = ROOT / "models" / "known_words.txt"
MIN_LEN_FOR_FUZZY = 4
MIN_KNOWN_COUNT = 3


def edit_distance(a, b):
    """Levenshtein distance with dynamic programming (insert = delete = substitute = 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,              # delete
                           cur[j - 1] + 1,           # insert
                           prev[j - 1] + (ca != cb)))  # substitute (0 if same letter)
        prev = cur
    return prev[-1]


class LexiconRule:
    name = "Lexicon rule"

    def __init__(self, words, known):
        self.words = set(words)
        self.known = set(known)
        self.by_length = {}
        for w in self.words:
            self.by_length.setdefault(len(w), []).append(w)
        self.is_bad = lru_cache(maxsize=None)(self._is_bad)

    @staticmethod
    def bad_words():
        return LEXICON_FILE.read_text(encoding="utf-8").split()

    @classmethod
    def build(cls, train_texts, train_labels):
        """Known words = words in 3+ clean training comments, minus the bad words."""
        from collections import Counter

        counts = Counter()
        for text, label in zip(train_texts, train_labels):
            if label == 0:
                counts.update(set(tokens(text)))
        bad = set(cls.bad_words())
        known = sorted(w for w, c in counts.items() if c >= MIN_KNOWN_COUNT and w not in bad)
        KNOWN_FILE.parent.mkdir(exist_ok=True)
        KNOWN_FILE.write_text("\n".join(known), encoding="utf-8")
        return cls(bad, known)

    @classmethod
    def load(cls):
        return cls(cls.bad_words(), KNOWN_FILE.read_text(encoding="utf-8").split())

    def _is_bad(self, word):
        if word in self.words:
            return True
        if word in self.known or len(word) < MIN_LEN_FOR_FUZZY:
            return False
        for n in (len(word) - 1, len(word), len(word) + 1):  # distance 1 => length differs by <= 1
            for w in self.by_length.get(n, []):
                if len(w) >= MIN_LEN_FOR_FUZZY and edit_distance(word, w) <= 1:
                    return True
        return False

    def matched_words(self, text):
        return [w for w in tokens(text) if self.is_bad(w)]

    def predict_proba(self, texts):
        return np.array([1.0 if any(self.is_bad(w) for w in tokens(t)) else 0.0 for t in texts])


if __name__ == "__main__":
    from src.data.load import load_split
    from src.eval.metrics import report

    model = LexiconRule.build(*load_split("train"))
    print(f"Bad words: {len(model.words)}, known words: {len(model.known):,}")
    texts, labels = load_split("val")
    report(model.name, labels, model.predict_proba(texts))
    print("Saved", KNOWN_FILE)
