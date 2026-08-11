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
