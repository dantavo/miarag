# tests/test_ingestion.py
from pathlib import Path
from miarag.ingestion import parse_report_text, ReportDoc

FIX = Path(__file__).parent / "fixtures" / "mini_report.txt"

def test_parse_extracts_company_and_person():
    raw = FIX.read_text()
    doc = parse_report_text(raw, doc_id="mini")
    assert isinstance(doc, ReportDoc)
    assert doc.company.startswith("ROSSI VERDI")
    assert doc.has_person is True          # la fixture ha CF esponenti
    assert "RSSMRA80A01F205X" not in doc.text  # pseudonimizzato

def test_parse_no_person():
    raw = "ACME SRL\nATTIVA\nSETTORE ATTIVITA'\nEdilizia\nEVENTI NEGATIVI\nAssenti"
    doc = parse_report_text(raw, doc_id="x")
    assert doc.has_person is False
