from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .config import PreprocessConfig


@dataclass(frozen=True)
class Token:
    text: str
    lemma: str
    pos: str | None
    start: int | None
    end: int | None


class TextPreprocessor:
    """Text preprocessing with optional NLP backends (spaCy > NLTK > regex).

    This class implements a practical preprocessing pipeline that honors
    :class:`PreprocessConfig` options such as:
      - lowercasing
      - stopword removal
      - POS filtering
      - minimum token length
      - optional lemmatization and POS tagging

    Backends:
      1) spaCy (preferred): tokenization, lemmatization, POS tagging
      2) NLTK fallback: regex tokenization with offsets + pos_tag + lemmatizer
      3) Regex-only fallback: lightweight rule-based POS + simple lemmatization

    Output is a list of :class:`Token` with stable character offsets.
    """

    WORD_RE = re.compile(r"[A-Za-z']+")

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.stopwords = self._load_stopwords()
        self._backend = self._init_backend()

    # ----------------------------
    # Resource loading
    # ----------------------------
    def _load_stopwords(self) -> set[str]:
        """Load stopwords from resources/stopwords.txt if present."""
        stopwords_path = Path(__file__).resolve().parent / "resources" / "stopwords.txt"
        if stopwords_path.exists():
            words = []
            for line in stopwords_path.read_text(encoding="utf-8").splitlines():
                w = line.strip()
                if w:
                    words.append(w.lower())
            return set(words)

        # Minimal fallback set (kept small on purpose)
        return {
            "the",
            "and",
            "a",
            "an",
            "in",
            "on",
            "of",
            "for",
            "to",
            "with",
            "is",
            "are",
        }

    # ----------------------------
    # Backend initialization
    # ----------------------------
    def _init_backend(self) -> dict[str, Any] | None:
        """Try to build an NLP backend.

        Returns a dict like:
          {"kind": "spacy", "nlp": <Language>}
          {"kind": "nltk", "pos_tag": <func>, "lemmatizer": <obj>}
        or None (regex fallback).
        """
        # If neither lemmatization nor POS tagging is requested, regex-only is enough.
        if not (self.cfg.pos_tag or self.cfg.lemmatize):
            return None

        # 1) spaCy
        backend = self._try_init_spacy()
        if backend is not None:
            return backend

        # 2) NLTK
        backend = self._try_init_nltk()
        if backend is not None:
            return backend

        return None

    def _try_init_spacy(self) -> dict[str, Any] | None:
        try:
            import spacy  # type: ignore
        except ImportError:
            return None

        # Prefer config language/model name; else fall back to en_core_web_sm; else blank.
        model_name = getattr(self.cfg, "language", "en") or "en"
        nlp = None
        try:
            nlp = spacy.load(model_name, disable=["parser", "ner"])
        except Exception:
            try:
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except Exception:
                try:
                    nlp = spacy.blank(model_name)
                except Exception:
                    return None

        # If we got a blank pipeline, lemmatizer may be missing; best-effort add.
        # (Not guaranteed to work without lookups, but harmless.)
        try:
            if nlp is not None and "lemmatizer" not in getattr(nlp, "pipe_names", []):
                nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
                nlp.initialize()
        except Exception:
            pass

        if nlp is None:
            return None
        return {"kind": "spacy", "nlp": nlp}

    def _try_init_nltk(self) -> dict[str, Any] | None:
        try:
            from nltk import pos_tag  # type: ignore
            from nltk.stem import WordNetLemmatizer  # type: ignore
        except Exception:
            return None

        # NLTK may fail at runtime if tagger corpora aren't downloaded.
        # We'll handle that in _process_nltk and fallback to regex.
        return {"kind": "nltk", "pos_tag": pos_tag, "lemmatizer": WordNetLemmatizer()}

    # ----------------------------
    # Public API
    # ----------------------------
    def process(self, text: str) -> list[Token]:
        """Process raw text and return filtered tokens."""
        if self._backend and self._backend.get("kind") == "spacy":
            return self._process_spacy(text)
        if self._backend and self._backend.get("kind") == "nltk":
            return self._process_nltk(text)
        return self._process_regex(text)

    # ----------------------------
    # Filtering helpers
    # ----------------------------
    def _pass_filters(self, lemma: str, pos: str | None) -> bool:
        if len(lemma) < self.cfg.min_token_len:
            return False
        if self.cfg.remove_stopwords and lemma in self.stopwords:
            return False
        if self.cfg.keep_pos and pos and pos not in self.cfg.keep_pos:
            return False
        return True

    # ----------------------------
    # spaCy backend
    # ----------------------------
    def _process_spacy(self, text: str) -> list[Token]:
        nlp = self._backend["nlp"]
        doc = nlp(text)

        out: list[Token] = []
        for tok in doc:
            raw = tok.text.lower() if self.cfg.lowercase else tok.text

            pos: str | None = tok.pos_ if self.cfg.pos_tag else None
            if pos == "":
                pos = None

            if self.cfg.lemmatize:
                lemma = tok.lemma_
                if not lemma or lemma == "-PRON-":
                    lemma = raw
            else:
                lemma = raw

            lemma = lemma.lower() if self.cfg.lowercase else lemma

            if not self._pass_filters(lemma, pos):
                continue

            out.append(Token(text=raw, lemma=lemma, pos=pos, start=tok.idx, end=tok.idx + len(tok.text)))

        return out

    # ----------------------------
    # NLTK backend (regex offsets + pos_tag + lemmatizer)
    # ----------------------------
    def _map_nltk_pos(self, tag: str) -> str:
        if tag.startswith("J"):
            return "ADJ"
        if tag.startswith("V"):
            return "VERB"
        if tag.startswith("R"):
            return "ADV"
        if tag.startswith("N"):
            return "NOUN"
        return "NOUN"

    def _wordnet_pos(self, coarse: str) -> str:
        return {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}.get(coarse, "n")

    def _process_nltk(self, text: str) -> list[Token]:
        matches = list(self.WORD_RE.finditer(text))
        raw_tokens = [m.group(0) for m in matches]

        # POS tagging may fail if NLTK resources aren't present → fallback to regex.
        try:
            if self.cfg.pos_tag:
                pos_tags = self._backend["pos_tag"](raw_tokens)
            else:
                pos_tags = [(t, "") for t in raw_tokens]
        except Exception:
            return self._process_regex(text)

        lemmatizer = self._backend["lemmatizer"]

        out: list[Token] = []
        for match, (tok_text, penn) in zip(matches, pos_tags):
            raw = tok_text.lower() if self.cfg.lowercase else tok_text
            pos: str | None = self._map_nltk_pos(penn) if self.cfg.pos_tag else None

            if self.cfg.lemmatize:
                try:
                    lemma = lemmatizer.lemmatize(raw, pos=self._wordnet_pos(pos or "NOUN"))
                except Exception:
                    lemma = raw
            else:
                lemma = raw

            lemma = lemma.lower() if self.cfg.lowercase else lemma

            if not self._pass_filters(lemma, pos):
                continue

            out.append(Token(text=raw, lemma=lemma, pos=pos, start=match.start(), end=match.end()))

        return out

    # ----------------------------
    # Regex-only fallback (simple POS + naive lemmatization)
    # ----------------------------
    def _simple_pos(self, token: str) -> str:
        lower = token.lower()
        if lower.endswith("ing") or lower.endswith("ed"):
            return "VERB"
        if lower.endswith("ly"):
            return "ADV"
        if lower.endswith(("ous", "ful", "able", "ive", "al")):
            return "ADJ"
        return "NOUN"

    def _simple_lemma(self, token: str, pos: str) -> str:
        if not self.cfg.lemmatize:
            return token.lower() if self.cfg.lowercase else token

        lower = token.lower()
        if pos == "VERB" and lower.endswith("ing") and len(lower) > 4:
            return lower[:-3]
        if pos == "VERB" and lower.endswith("ed") and len(lower) > 3:
            return lower[:-2]
        if pos == "NOUN" and lower.endswith("s") and len(lower) > 3:
            return lower[:-1]
        return lower

    def _process_regex(self, text: str) -> list[Token]:
        out: list[Token] = []
        for match in self.WORD_RE.finditer(text):
            raw_text = match.group(0)
            raw = raw_text.lower() if self.cfg.lowercase else raw_text

            pos: str | None = self._simple_pos(raw) if self.cfg.pos_tag else None
            lemma = self._simple_lemma(raw, pos or "NOUN")

            if not self._pass_filters(lemma, pos):
                continue

            out.append(Token(text=raw, lemma=lemma, pos=pos, start=match.start(), end=match.end()))
        return out


__all__ = ["Token", "TextPreprocessor"]
