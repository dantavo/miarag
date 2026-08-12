import csv
from pathlib import Path
from miarag.dashboard_helpers import (
    load_summary_rows, pii_demo, live_s2mia_on_chunk, live_budgetleak_on_chunk,
)
from miarag.rag import RAGResponse

class _FakeRAG:
    def query(self, q, max_tokens=256):
        return RAGResponse(answer=q, retrieved_ids=["c1", "c2"], perplexity=None)
    def perplexity_of(self, text):
        return 3.0

def test_load_summary_rows_missing_returns_empty(tmp_path):
    assert load_summary_rows(tmp_path / "nope.csv") == []

def test_load_summary_rows_reads_all(tmp_path):
    p = tmp_path / "summary.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["attack", "subgroup", "auc", "tpr_at_1fpr", "ppv", "advantage"])
        w.writeheader()
        w.writerow({"attack": "s2mia", "subgroup": "all", "auc": "0.75",
                    "tpr_at_1fpr": "0.4", "ppv": "0.8", "advantage": "0.3"})
    rows = load_summary_rows(p)
    assert len(rows) == 1 and rows[0]["attack"] == "s2mia"

def test_pii_demo_pseudonymizes_cf():
    # regex-only demo catches CF/PIVA/REA, not email (NER-only)
    out = pii_demo("CF RSSMRA80A01F205X presente")
    assert "RSSMRA80A01F205X" not in out
    assert "CF_" in out

def test_live_s2mia_on_chunk_returns_features_and_score():
    r = live_s2mia_on_chunk(_FakeRAG(), "questo e' un chunk di prova con parole varie")
    assert "bleu" in r and "perplexity" in r and "score" in r
    assert isinstance(r["score"], float)

def test_live_budgetleak_on_chunk_returns_seq_and_score():
    r = live_budgetleak_on_chunk(_FakeRAG(), "questo e' un chunk di prova con parole varie")
    assert "sequence" in r and isinstance(r["sequence"], list)
    assert "score" in r and isinstance(r["score"], float)
