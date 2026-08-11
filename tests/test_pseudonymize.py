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
    """Detector fittizio: marca 'Mario Rossi' come PER, senza caricare modelli."""
    def detect(self, text):
        spans = []
        needle = "Mario Rossi"
        i = text.find(needle)
        if i >= 0:
            spans.append((i, i + len(needle), "PER"))
        return spans

def test_ner_replaces_person_name_in_prose():
    t = "Il legale rappresentante Mario Rossi ha firmato."
    out = pseudonymize_text(t, ner=_FakeNer())
    assert "Mario Rossi" not in out
    assert "PER_" in out

def test_regex_wins_on_overlap():
    # un CF non deve essere spezzato da uno span NER sovrapposto
    class _Greedy:
        def detect(self, text): return [(0, len(text), "PER")]
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
