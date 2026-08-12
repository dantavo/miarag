# tests/test_ingestion_loaders.py
import json
from pathlib import Path
from miarag.ingestion import load_md, load_txt, load_any, ingest_file, ReportDoc

class _FakeNer:
    def detect(self, text): return []   # nessuna PII, has_person=False

def test_load_txt_and_md(tmp_path):
    t = tmp_path / "a.txt"; t.write_text("Azienda X\nfatturato 100", encoding="utf-8")
    m = tmp_path / "b.md"; m.write_text("# Titolo\n\nCorpo del documento.", encoding="utf-8")
    assert "fatturato 100" in load_txt(t)
    assert "Corpo del documento" in load_md(m)

def test_load_any_dispatches_by_suffix(tmp_path):
    t = tmp_path / "a.TXT"; t.write_text("ciao", encoding="utf-8")   # suffix case-insensitive
    assert load_any(t).strip() == "ciao"

def test_load_any_rejects_unknown(tmp_path):
    bad = tmp_path / "x.zip"; bad.write_bytes(b"PK\x03\x04")
    try:
        load_any(bad); assert False, "atteso ValueError"
    except ValueError:
        pass

def test_ingest_file_appends_one_report(tmp_path):
    out = tmp_path / "reports.jsonl"
    f1 = tmp_path / "doc1.txt"; f1.write_text("Prima riga\nresto", encoding="utf-8")
    f2 = tmp_path / "doc2.md"; f2.write_text("Seconda\nresto", encoding="utf-8")
    r1 = ingest_file(f1, out, ner=_FakeNer(), append=False)   # crea il file
    r2 = ingest_file(f2, out, ner=_FakeNer(), append=True)    # appende
    assert isinstance(r1, ReportDoc) and isinstance(r2, ReportDoc)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    d0 = json.loads(lines[0]); d1 = json.loads(lines[1])
    assert d0["doc_id"] == "doc1" and d1["doc_id"] == "doc2"
    assert set(d0.keys()) == {"doc_id", "company", "text", "has_person"}
