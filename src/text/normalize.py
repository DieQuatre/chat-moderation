"""Regex normalization of noisy chat text (technique area B).

Steps, in order:
  1. lowercase
  2. remove URLs and IP addresses
  3. undo leetspeak inside words that contain letters   (y0u @re -> you are)
  4. join letters split by separators                    (i.d.i.o.t -> idiot)
  5. shorten letters repeated 3+ times to 2               (stuuuupid -> stuupid)
  6. collapse whitespace
Every model gets the same normalized text.
"""
import re

URL = re.compile(r"https?://\S+|www\.\S+")
IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
WORD = re.compile(r"\S+")
HAS_LETTER = re.compile(r"[a-z]")
LEET = str.maketrans({"@": "a", "4": "a", "3": "e", "1": "i", "0": "o", "$": "s", "5": "s", "7": "t"})
SPLIT_LETTERS = re.compile(r"\b(?:[a-z][.*_\-]){2,}[a-z]\b")
SEPARATORS = re.compile(r"[.*_\-]")
REPEATS = re.compile(r"([a-z])\1{2,}")
SPACES = re.compile(r"\s+")
LETTER_WORDS = re.compile(r"[a-z]+")


def _deleet(match):
    word = match.group(0)
    return word.translate(LEET) if HAS_LETTER.search(word) else word


def normalize(text):
    text = str(text).lower()
    text = URL.sub(" ", text)
    text = IP.sub(" ", text)
    text = WORD.sub(_deleet, text)
    text = SPLIT_LETTERS.sub(lambda m: SEPARATORS.sub("", m.group(0)), text)
    text = REPEATS.sub(r"\1\1", text)
    return SPACES.sub(" ", text).strip()


def tokens(text):
    """Normalized text -> list of words (letters only)."""
    return LETTER_WORDS.findall(normalize(text))


if __name__ == "__main__":
    examples = [
        "Y0U @RE S7UP1D!!!",
        "what an 1d10t",
        "you are such an I.D.I.O.T",
        "stuuuuupid noooob",
        "see http://example.com/page or www.example.org",
        "Edited by 192.168.0.1 in 2018,\n\nthanks for the 100 edits",
        "my e-mail is fine",
    ]
    for text in examples:
        print(f"{text!r:60} -> {normalize(text)!r}")
