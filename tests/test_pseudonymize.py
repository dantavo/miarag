# tests/test_pseudonymize.py
import re
from miarag.pseudonymize import pseudonymize_text, token_for, regex_spans

def test_cf_replaced_and_deterministic():
    t = "AMMINISTRATORE CF RSSMRA80A01F205X presente"
    out1 = pseudonymize_text(t)          # ner=None → solo regex
    out2 = pseudonymize_text(t)
    assert "RSSMRA80A01F205X" not in out1
    assert "CF_" in out1
    assert out1 == out2                  # deterministico

def test_piva_replaced():
    out = pseudonymize_text("P.IVA 01234567890 qui")
    assert "01234567890" not in out
    assert "PIVA_" in out

def test_same_id_same_token():
    out = pseudonymize_text("CF RSSMRA80A01F205X e ancora RSSMRA80A01F205X")
    tokens = [w for w in out.split() if w.startswith("CF_")]
    assert len(tokens) == 2 and tokens[0] == tokens[1]

def test_token_stable_across_calls():
    assert token_for("PER", "Mario Rossi") == token_for("PER", "Mario Rossi")

class _FakeNer:
    """Detector fittizio: marca 'Mario Rossi' come PERSON (canonical NER label mapping), senza caricare modelli."""
    def detect(self, text):
        spans = []
        needle = "Mario Rossi"
        i = text.find(needle)
        if i >= 0:
            spans.append((i, i + len(needle), "PERSON"))
        return spans

def test_ner_replaces_person_name_in_prose():
    t = "Il legale rappresentante Mario Rossi ha firmato."
    out = pseudonymize_text(t, ner=_FakeNer())
    assert "Mario Rossi" not in out
    assert "PERSON_" in out

def test_regex_wins_on_overlap():
    # un CF non deve essere spezzato da uno span NER sovrapposto
    class _Greedy:
        def detect(self, text): return [(0, len(text), "PERSON")]
    out = pseudonymize_text("RSSMRA80A01F205X", ner=_Greedy())
    assert out.startswith("CF_")

def test_unmapped_label_logged(caplog):
    """Carry-over from Task 1: unmapped entity_group labels must be logged, not silently dropped."""
    from miarag.pseudonymize import ItalianPIINerDetector
    import logging

    # Mock pipeline that returns an unknown label
    class _FakePipeline:
        def __call__(self, text):
            return [{"entity_group": "UNKNOWN_ORG", "start": 0, "end": 4, "score": 0.99}]

    detector = ItalianPIINerDetector()
    detector._pipe = _FakePipeline()  # bypass lazy-load

    with caplog.at_level(logging.WARNING, logger="miarag.pseudonymize"):
        spans = detector.detect("test")

    # The span is dropped (not in _LABEL_MAP)
    assert spans == []
    # But a warning is logged
    assert "Unmapped entity_group from NER model: 'UNKNOWN_ORG'" in caplog.text

    # Second call with same label should not log again (dedupe via _warned_labels)
    caplog.clear()
    spans2 = detector.detect("test")
    assert spans2 == []
    assert "UNKNOWN_ORG" not in caplog.text  # no duplicate warning

class _FakeRealLabels:
    """Fake NER that mimics ItalianPIINerDetector mapping behavior.
    Returns mapped kind strings (PERSON, EMAIL, etc.) that pseudonymize_text expects.
    For unmapped labels (ORG, CITY, etc.), they would be filtered out by ItalianPIINerDetector.detect(),
    so this fake doesn't return them at all — simulating the detector's behavior.
    """
    def detect(self, text):
        spans = []
        # Pseudonymize targets — return the MAPPED kind (what ItalianPIINerDetector would return)
        if "mario.rossi@example.it" in text:
            i = text.find("mario.rossi@example.it")
            spans.append((i, i + 22, "EMAIL"))
        if "Mario Rossi" in text:
            i = text.find("Mario Rossi")
            spans.append((i, i + 11, "PERSON"))
        if "IT60X0542811101000000123456" in text:
            i = text.find("IT60X0542811101000000123456")
            spans.append((i, i + 27, "IBAN"))
        if "3391234567" in text:
            i = text.find("3391234567")
            spans.append((i, i + 10, "PHONE"))
        if "Via Roma" in text:
            i = text.find("Via Roma")
            spans.append((i, i + 8, "STREET"))
        if "20100" in text:
            i = text.find("20100")
            spans.append((i, i + 5, "ZIP"))
        # Keep targets (ORG, CITY, DATE, AMOUNT) are NOT returned here —
        # ItalianPIINerDetector filters them out since they're not in _LABEL_MAP
        return spans

def test_real_labels_pseudonymize_identifiers():
    """Direct identifiers (FULLNAME, EMAIL, IBAN, PHONE, etc.) get replaced."""
    t = "Mario Rossi mario.rossi@example.it IBAN IT60X0542811101000000123456 tel 3391234567 Via Roma 20100"
    out = pseudonymize_text(t, ner=_FakeRealLabels())
    assert "Mario Rossi" not in out
    assert "mario.rossi@example.it" not in out
    assert "IT60X0542811101000000123456" not in out
    assert "3391234567" not in out
    assert "Via Roma" not in out
    assert "20100" not in out
    # Check tokens are present
    assert "PERSON_" in out
    assert "EMAIL_" in out
    assert "IBAN_" in out
    assert "PHONE_" in out
    assert "STREET_" in out
    assert "ZIP_" in out

def test_real_labels_keep_content():
    """Content labels (CITY, DATE, AMOUNT) are NOT pseudonymized.
    Note: ORG is now mapped to COMPANY (v0.2 fix — company names ARE direct
    identifiers per GDPR Art. 4(1)). See test_org_pseudonymized_as_company."""
    t = "QUANTYCA S.P.A. Milano 12/03/1980 500000 euro"
    out = pseudonymize_text(t, ner=_FakeRealLabels())
    # These must remain intact
    assert "QUANTYCA S.P.A." in out    # _FakeRealLabels doesn't return ORG span
    assert "Milano" in out
    assert "12/03/1980" in out
    assert "500000" in out
    # No tokens created for them
    assert "CITY_" not in out
    assert "DATE_" not in out
    assert "AMOUNT_" not in out

def test_expected_unmapped_labels_not_logged(caplog):
    """CITY, DATE, AMOUNT, AGE, PROVINCE are expected unmapped — no warning.
    Note: ORG is intentionally REMOVED from this list — v0.2 maps ORG→COMPANY."""
    from miarag.pseudonymize import ItalianPIINerDetector
    import logging

    class _FakePipeline:
        def __call__(self, text):
            return [
                {"entity_group": "CITY", "start": 5, "end": 10, "score": 0.99},
                {"entity_group": "DATE", "start": 11, "end": 15, "score": 0.99},
                {"entity_group": "AMOUNT", "start": 16, "end": 20, "score": 0.99},
                {"entity_group": "AGE", "start": 21, "end": 23, "score": 0.99},
                {"entity_group": "PROVINCE", "start": 24, "end": 26, "score": 0.99},
            ]

    detector = ItalianPIINerDetector()
    detector._pipe = _FakePipeline()

    with caplog.at_level(logging.WARNING, logger="miarag.pseudonymize"):
        spans = detector.detect("test")

    # All are dropped (expected unmapped)
    assert spans == []
    # No warning logged for expected labels
    assert "CITY" not in caplog.text
    assert "DATE" not in caplog.text
    assert "AMOUNT" not in caplog.text
    assert "AGE" not in caplog.text
    assert "PROVINCE" not in caplog.text


def test_org_pseudonymized_as_company():
    """ORG label from NER → COMPANY token. Enterprise names are direct PII (GDPR)."""
    from miarag.pseudonymize import ItalianPIINerDetector

    class _FakePipeline:
        def __call__(self, text):
            # Simulate NER detecting a company name as ORG
            i = text.find("Acme")
            if i >= 0:
                return [{"entity_group": "ORG", "start": i, "end": i + 4, "score": 0.99}]
            return []

    detector = ItalianPIINerDetector()
    detector._pipe = _FakePipeline()

    spans = detector.detect("Il gruppo Acme opera in Italia")
    assert spans == [(10, 14, "COMPANY")]

    # End-to-end via pseudonymize_text
    out = pseudonymize_text("Il gruppo Acme opera in Italia", ner=detector)
    assert "Acme" not in out
    assert "COMPANY_" in out


def test_company_regex_from_env(monkeypatch):
    """COMPANY regex is built from env MIARAG_COMPANY_NAMES (no hard-coded brand).
    Fail-safe if NER misses a company name."""
    monkeypatch.setenv("MIARAG_COMPANY_NAMES", "Acme Corp,Acme Corp S.p.A.,acme.com")
    variants = [
        "Acme Corp",
        "Acme Corp S.p.A.",
        "acme.com",
    ]
    for v in variants:
        out = pseudonymize_text(f"Il testo cita {v} come fornitore.")
        assert v not in out, f"company variant leaked: {v} in {out!r}"
        assert "COMPANY_" in out


def test_company_regex_absent_when_env_unset(monkeypatch):
    """No env → no hard-coded company detection (relies on NER only)."""
    monkeypatch.delenv("MIARAG_COMPANY_NAMES", raising=False)
    out = pseudonymize_text("Il testo cita Acme Corp come fornitore.")
    # Without env config and without NER, the company name is NOT pseudonymized
    # by regex (only CF/PIVA/REA/long-digits are). This is expected: brand names
    # are runtime config, not source constants.
    assert "Acme Corp" in out


def test_rizzo_ner_detector_backcompat_alias():
    """RizzoNerDetector alias to ItalianPIINerDetector (v0.2 backcompat)."""
    from miarag.pseudonymize import RizzoNerDetector, ItalianPIINerDetector
    assert RizzoNerDetector is ItalianPIINerDetector

def test_genuinely_unknown_label_still_warns(caplog):
    """A truly unknown label (not in map, not expected) still triggers warning."""
    from miarag.pseudonymize import ItalianPIINerDetector
    import logging

    class _FakePipeline:
        def __call__(self, text):
            return [{"entity_group": "WEIRDLABEL", "start": 0, "end": 4, "score": 0.99}]

    detector = ItalianPIINerDetector()
    detector._pipe = _FakePipeline()

    with caplog.at_level(logging.WARNING, logger="miarag.pseudonymize"):
        spans = detector.detect("test")

    assert spans == []
    assert "Unmapped entity_group from NER model: 'WEIRDLABEL'" in caplog.text

class _OverlapNer:
    """Fake NER returning overlapping spans like a real Italian NER model."""
    def detect(self, text):
        spans = []
        # Repro case: '1.045    13690054749    P.IVA 01234567890 tel 3391234567'
        # NER produces overlapping/nested spans
        if "1.045" in text and "13690054749" in text:
            # Spurious 1-char BUILDINGNUM nested inside "1.045"
            i = text.find("1.045")
            spans.append((i+2, i+3, "BUILDINGNUM"))  # '0' at offset 2-3
        if "13690054749" in text:
            # Off-by-one overlap: NER sees partial PIVA
            i = text.find("13690054749")
            spans.append((i+1, i+11, "PIVA"))  # '3690054749' - one char off from regex match
        if "01234567890" in text:
            i = text.find("01234567890")
            # NER with leading space
            spans.append((i-1, i+11, "PIVA"))  # ' 01234567890'
        if "3391234567" in text:
            i = text.find("3391234567")
            # NER with leading space
            spans.append((i-1, i+10, "PHONE"))  # ' 3391234567'
        return spans

def test_overlapping_ner_vs_regex_piva():
    """Off-by-one NER-vs-regex PIVA overlap → exactly ONE token, no leftover digits."""
    # Real regex will match the full 11-digit PIVA
    t = "1.045    13690054749    P.IVA 01234567890 tel 3391234567"
    out = pseudonymize_text(t, ner=_OverlapNer())

    # The full 11-digit PIVAs must be tokenized (regex wins)
    assert "13690054749" not in out
    assert "01234567890" not in out
    assert "3391234567" not in out

    # Must have PIVA and PHONE tokens
    assert "PIVA_" in out
    assert "PHONE_" in out

    # CRITICAL: no 9+ digit run may remain in cleartext (would be a PIVA/phone leak)
    assert re.search(r'\d{9,}', out) is None, f"Found 9+ digit run in: {out}"

def test_nested_spurious_buildingnum_dropped():
    """Nested 1-char BUILDINGNUM inside a number → dropped, surrounding content intact."""
    t = "1.045    13690054749    P.IVA 01234567890 tel 3391234567"
    out = pseudonymize_text(t, ner=_OverlapNer())

    # The spurious BUILDINGNUM at offset 2-3 ('0' inside "1.045") must be dropped
    # because the regex PIVA span or the longer context should win
    # Result: "1.045" either stays intact OR the whole number gets tokenized once
    # but "1.045" must NOT become "1.BUILDINGNUM_..45"
    assert "1.BUILDINGNUM_" not in out, f"Spurious nested BUILDINGNUM corrupted content: {out}"

    # The number should either be intact or fully replaced by a PIVA token, not split
    # In this case "1.045" is separate from the PIVA, so it should stay intact
    if "PIVA_" in out:
        # Check that we don't have partial corruption
        assert ".BUILDINGNUM_" not in out

def test_creditcard_and_catasto_mapped():
    """CREDITCARDNUMBER e CATASTO (scoperti sui PDF reali) devono essere pseudonimizzati, non skippati."""
    from miarag.pseudonymize import ItalianPIINerDetector
    class _FakePipeline:
        def __call__(self, text):
            return [
                {"entity_group": "CREDITCARDNUMBER", "start": 0, "end": 4, "score": 0.99},
                {"entity_group": "CATASTO", "start": 5, "end": 9, "score": 0.99},
            ]
    det = ItalianPIINerDetector()
    det._pipe = _FakePipeline()
    spans = det.detect("1234 5678")
    kinds = sorted(k for (_, _, k) in spans)
    assert kinds == ["CATASTO", "CC"]

def test_no_digit_leak_across_split_number():
    """Bug reale: numeri di bilancio spezzati (nbsp) → NER tagga pezzi non contigui;
    la ricostruzione per concatenazione non deve MAI riassemblare cifre in chiaro."""
    # simula pezzi non contigui: NER marca due sottostringhe separate da spazio
    class _SplitNer:
        def detect(self, text):
            spans = []
            # marca solo il primo blocco di 4 cifre come PHONE, lascia il resto
            i = text.find("1369")
            if i >= 0:
                spans.append((i, i + 4, "PHONE"))
            return spans
    t = "saldo 1369 0054749 fine"
    out = pseudonymize_text(t, ner=_SplitNer())
    # il pezzo taggato è tokenizzato, il resto resta com'era (NON deve incollarsi al token)
    assert "PHONE_" in out
    # token ben formato (8 hex), nessuna cifra reale appiccicata al prefisso
    assert re.search(r"PHONE_[0-9a-f]{8}", out)
    assert re.search(r"PHONE_\d{5,}", out) is None, f"cifre reali incollate al token: {out}"

def test_long_digit_backstop_no_ner_no_regex():
    """Fail-closed: una sequenza ≥9 cifre non catturata da NER né regex viene comunque tokenizzata."""
    out = pseudonymize_text("id interno 936795600 nel testo")  # ner=None, non è PIVA(11)/REA(lettere)
    assert "936795600" not in out
    assert re.search(r"NUM_[0-9a-f]{8}", out)
    assert re.search(r"\d{9,}", out) is None

def test_ner_vs_ner_overlap():
    """Two overlapping NER spans → only one accepted, no overlap in output."""
    class _DoubleNer:
        def detect(self, text):
            # Two overlapping PERSON spans
            if "Mario Rossi Bianchi" in text:
                i = text.find("Mario Rossi Bianchi")
                return [
                    (i, i+11, "PERSON"),      # "Mario Rossi"
                    (i+6, i+19, "PERSON"),    # "Rossi Bianchi" - overlaps with first
                ]
            return []

    t = "Il sig. Mario Rossi Bianchi è presente."
    out = pseudonymize_text(t, ner=_DoubleNer())

    # Only one PERSON token should appear (the first/longer one wins)
    person_tokens = [w for w in out.split() if "PERSON_" in w]
    assert len(person_tokens) == 1, f"Expected 1 PERSON token, got {len(person_tokens)}: {out}"

    # The overlapping region should not be double-tokenized
    assert out.count("PERSON_") == 1
