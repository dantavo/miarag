# src/miarag/pseudonymize.py
import hashlib
import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

PII_PATTERNS = {
    "CF": re.compile(r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b"),
    "PIVA": re.compile(r"\b\d{11}\b"),
    "REA": re.compile(r"\b[A-Z]{2}\d{6,7}\b"),
}

def token_for(kind: str, value: str, seed: int = 42) -> str:
    h = hashlib.sha256(f"{seed}:{kind}:{value}".encode()).hexdigest()[:8]
    return f"{kind}_{h}"

def regex_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    for kind, pat in PII_PATTERNS.items():
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), kind))
    return spans

class NerDetector(Protocol):
    def detect(self, text: str) -> list[tuple[int, int, str]]: ...

def _merge(regex, ner):
    """Regex ha priorità: scarta span NER che si sovrappongono a uno regex."""
    out = list(regex)
    for (s, e, k) in ner:
        if any(not (e <= rs or s >= re_) for (rs, re_, _) in regex):
            continue
        out.append((s, e, k))
    return sorted(out, key=lambda x: x[0])

def pseudonymize_text(text: str, seed: int = 42, ner: NerDetector | None = None) -> str:
    rx = regex_spans(text)
    nr = ner.detect(text) if ner is not None else []
    spans = _merge(rx, nr)
    for (s, e, kind) in reversed(spans):           # da destra: gli indici non slittano
        text = text[:s] + token_for(kind, text[s:e], seed) + text[e:]
    return text

class RizzoNerDetector:
    """NER italiano rizzo-pii-0.3B via transformers. Lazy-load; solo PER/LOC."""
    _LABEL_MAP = {"PER": "PER", "PERSON": "PER", "FULLNAME": "PER",
                  "LOC": "LOC", "ADDRESS": "LOC", "GPE": "LOC"}

    def __init__(self, model_id: str = "rizzoaiacademy/rizzo-pii-0.3B"):
        self._model_id = model_id
        self._pipe = None
        self._warned_labels = set()  # track unmapped labels already warned

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("token-classification", model=self._model_id,
                                  aggregation_strategy="simple")

    def detect(self, text: str) -> list[tuple[int, int, str]]:
        self._ensure()
        spans = []
        for ent in self._pipe(text):
            grp = str(ent.get("entity_group", "")).upper()
            kind = self._LABEL_MAP.get(grp)
            if kind:
                spans.append((int(ent["start"]), int(ent["end"]), kind))
            elif grp and grp not in self._warned_labels:
                # Defensive: log unmapped labels once to surface silent drops
                logger.warning(f"Unmapped entity_group from NER model: '{grp}' (span skipped)")
                self._warned_labels.add(grp)
        return spans
