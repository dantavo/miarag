# tests/test_ingestion.py
from pathlib import Path
from miarag.ingestion import parse_report_text, ReportDoc

FIX = Path(__file__).parent / "fixtures" / "mini_report.txt"

class _FakeNerWithPerson:
    """Fake NER that detects FULLNAME (person names) to trigger has_person=True."""
    def detect(self, text):
        spans = []
        # Look for person names in the test fixture
        for name in ["MARIO ROSSI", "GIULIA VERDI"]:
            i = text.find(name)
            if i >= 0:
                spans.append((i, i + len(name), "PERSON"))
        # Also mark CF spans so they get pseudonymized
        import re
        cf_pat = re.compile(r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b")
        for m in cf_pat.finditer(text):
            spans.append((m.start(), m.end(), "CF"))
        return spans

class _FakeNerNoPerson:
    """Fake NER that detects no FULLNAME spans → has_person=False."""
    def detect(self, text):
        return []

def test_parse_extracts_company_and_person():
    raw = FIX.read_text()
    doc = parse_report_text(raw, doc_id="mini", ner=_FakeNerWithPerson())
    assert isinstance(doc, ReportDoc)
    assert doc.company.startswith("ROSSI VERDI")
    assert doc.has_person is True          # NER found FULLNAME spans
    assert "RSSMRA80A01F205X" not in doc.text  # pseudonimizzato by fake NER

def test_parse_no_person():
    raw = "ACME SRL\nATTIVA\nSETTORE ATTIVITA'\nEdilizia\nEVENTI NEGATIVI\nAssenti"
    doc = parse_report_text(raw, doc_id="x", ner=_FakeNerNoPerson())
    assert doc.has_person is False  # NER found no FULLNAME
