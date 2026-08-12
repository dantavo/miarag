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

_LONG_DIGITS = re.compile(r"\d{9,}")

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

def _merge(regex, ner, text: str = ""):
    """Resolve ALL overlaps (regex-vs-NER, NER-vs-NER, nested) with two-pass approach.
    Regex has ABSOLUTE priority (accepted first); then NER spans that don't overlap.
    Within each pass, longer spans win over shorter; no overlaps remain.

    Also filters out pathologically short (1-2 char) numeric BUILDINGNUM spans that sit
    inside longer digit runs — these are spurious model tags that corrupt content.
    """
    # Filter out spurious short numeric BUILDINGNUM spans embedded in digit runs
    filtered_ner = []
    for (s, e, k) in ner:
        # Drop 1-2 char BUILDINGNUM spans that are purely numeric and surrounded by digits
        if k == "BUILDINGNUM" and (e - s) <= 2 and text:
            span_text = text[s:e]
            if span_text.isdigit():
                # Check if surrounded by digits (part of a larger number)
                before = text[s-1:s] if s > 0 else ""
                after = text[e:e+1] if e < len(text) else ""
                if before.isdigit() or after.isdigit() or before == "." or after == ".":
                    # Skip this span - it's a spurious tag inside a larger number
                    continue
        filtered_ner.append((s, e, k))

    # Two-pass approach: regex first (absolute priority), then NER
    accepted = []

    # Pass 1: Accept all non-overlapping regex spans (sort by start, then longer first)
    regex_sorted = sorted(regex, key=lambda x: (x[0], -(x[1] - x[0])))
    for (s, e, k) in regex_sorted:
        if any(not (e <= acc_s or s >= acc_e) for (acc_s, acc_e, _) in accepted):
            continue
        accepted.append((s, e, k))

    # Pass 2: Accept non-overlapping NER spans (sort by start, then longer first)
    ner_sorted = sorted(filtered_ner, key=lambda x: (x[0], -(x[1] - x[0])))
    for (s, e, k) in ner_sorted:
        # Check overlap with ALL accepted spans (regex + already-accepted NER)
        if any(not (e <= acc_s or s >= acc_e) for (acc_s, acc_e, _) in accepted):
            continue
        accepted.append((s, e, k))

    return sorted(accepted, key=lambda x: x[0])

def pseudonymize_text(text: str, seed: int = 42, ner: NerDetector | None = None) -> str:
    rx = regex_spans(text)
    nr = ner.detect(text) if ner is not None else []
    spans = _merge(rx, nr, text)
    # Ricostruzione per concatenazione di slice tra span NON sovrapposti: immune
    # a slittamenti d'indice (la mutazione in-place corrompeva testi con span
    # residui adiacenti / numeri spezzati da nbsp).
    out = []
    prev = 0
    for (s, e, kind) in spans:
        if s < prev:            # overlap residuo: salta (già coperto)
            continue
        out.append(text[prev:s])

        # Preserva whitespace leading/trailing: se span inizia/finisce con spazio,
        # includi spazio prima/dopo token (NER rizzo spesso cattura whitespace)
        span_text = text[s:e]
        leading_ws = ""
        trailing_ws = ""
        if span_text and span_text[0].isspace():
            leading_ws = span_text[0]
            span_text = span_text[1:]
            s += 1
        if span_text and span_text[-1].isspace():
            trailing_ws = span_text[-1]
            span_text = span_text[:-1]
            e -= 1

        out.append(leading_ws + token_for(kind, span_text, seed) + trailing_ws)
        prev = e
    out.append(text[prev:])
    result = "".join(out)
    # Backstop fail-closed: qualunque sequenza di ≥9 cifre sfuggita a NER+regex
    # (PIVA/REA/telefoni/carte spezzati da separatori unicode, id vari) viene
    # comunque tokenizzata. Il gate etico non deve mai far uscire cifre lunghe in chiaro.
    result = _LONG_DIGITS.sub(lambda m: token_for("NUM", m.group(), seed), result)
    return result

class RizzoNerDetector:
    """NER italiano rizzo-pii-0.3B via transformers. Lazy-load; maps real rizzo-pii labels."""
    _LABEL_MAP = {
        "FULLNAME": "PERSON",
        "CF": "CF",
        "PIVA": "PIVA",
        "IBAN": "IBAN",
        "EMAIL": "EMAIL",
        "TELEPHONENUM": "PHONE",
        "STREET": "STREET",
        "BUILDINGNUM": "BUILDINGNUM",
        "ZIPCODE": "ZIP",
        "ID_DOC": "DOCID",
        "DOCID": "DOCID",
        "TARGA": "PLATE",
        "CREDITCARDNUMBER": "CC",
        "CATASTO": "CATASTO",
    }
    _EXPECTED_UNMAPPED = {"ORG", "CITY", "PROVINCE", "DATE", "AMOUNT", "AGE", "TIME"}

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
            elif grp and grp not in self._EXPECTED_UNMAPPED and grp not in self._warned_labels:
                # Defensive: log unmapped labels once to surface silent drops
                # (expected keep-labels like ORG/CITY/DATE are not logged)
                logger.warning(f"Unmapped entity_group from NER model: '{grp}' (span skipped)")
                self._warned_labels.add(grp)
        return spans
