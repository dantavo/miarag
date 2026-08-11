# tests/test_pseudonymize.py
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
    """Detector fittizio: marca 'Mario Rossi' come PERSON (real rizzo-pii label), senza caricare modelli."""
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
    from miarag.pseudonymize import RizzoNerDetector
    import logging

    # Mock pipeline that returns an unknown label
    class _FakePipeline:
        def __call__(self, text):
            return [{"entity_group": "UNKNOWN_ORG", "start": 0, "end": 4, "score": 0.99}]

    detector = RizzoNerDetector()
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
    """Fake NER that mimics RizzoNerDetector mapping behavior.
    Returns mapped kind strings (PERSON, EMAIL, etc.) that pseudonymize_text expects.
    For unmapped labels (ORG, CITY, etc.), they would be filtered out by RizzoNerDetector.detect(),
    so this fake doesn't return them at all — simulating the detector's behavior.
    """
    def detect(self, text):
        spans = []
        # Pseudonymize targets — return the MAPPED kind (what RizzoNerDetector would return)
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
        # RizzoNerDetector filters them out since they're not in _LABEL_MAP
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
    """Content labels (ORG, CITY, DATE, AMOUNT) are NOT pseudonymized."""
    t = "QUANTYCA S.P.A. Milano 12/03/1980 500000 euro"
    out = pseudonymize_text(t, ner=_FakeRealLabels())
    # These must remain intact
    assert "QUANTYCA S.P.A." in out
    assert "Milano" in out
    assert "12/03/1980" in out
    assert "500000" in out
    # No tokens created for them
    assert "ORG_" not in out
    assert "CITY_" not in out
    assert "DATE_" not in out
    assert "AMOUNT_" not in out

def test_expected_unmapped_labels_not_logged(caplog):
    """ORG, CITY, DATE, AMOUNT, AGE, PROVINCE are expected unmapped — no warning."""
    from miarag.pseudonymize import RizzoNerDetector
    import logging

    class _FakePipeline:
        def __call__(self, text):
            return [
                {"entity_group": "ORG", "start": 0, "end": 4, "score": 0.99},
                {"entity_group": "CITY", "start": 5, "end": 10, "score": 0.99},
                {"entity_group": "DATE", "start": 11, "end": 15, "score": 0.99},
                {"entity_group": "AMOUNT", "start": 16, "end": 20, "score": 0.99},
                {"entity_group": "AGE", "start": 21, "end": 23, "score": 0.99},
                {"entity_group": "PROVINCE", "start": 24, "end": 26, "score": 0.99},
            ]

    detector = RizzoNerDetector()
    detector._pipe = _FakePipeline()

    with caplog.at_level(logging.WARNING, logger="miarag.pseudonymize"):
        spans = detector.detect("test")

    # All are dropped (expected unmapped)
    assert spans == []
    # No warning logged for expected labels
    assert "ORG" not in caplog.text
    assert "CITY" not in caplog.text
    assert "DATE" not in caplog.text
    assert "AMOUNT" not in caplog.text
    assert "AGE" not in caplog.text
    assert "PROVINCE" not in caplog.text

def test_genuinely_unknown_label_still_warns(caplog):
    """A truly unknown label (not in map, not expected) still triggers warning."""
    from miarag.pseudonymize import RizzoNerDetector
    import logging

    class _FakePipeline:
        def __call__(self, text):
            return [{"entity_group": "WEIRDLABEL", "start": 0, "end": 4, "score": 0.99}]

    detector = RizzoNerDetector()
    detector._pipe = _FakePipeline()

    with caplog.at_level(logging.WARNING, logger="miarag.pseudonymize"):
        spans = detector.detect("test")

    assert spans == []
    assert "Unmapped entity_group from NER model: 'WEIRDLABEL'" in caplog.text
