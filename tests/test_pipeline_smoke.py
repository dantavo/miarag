# tests/test_pipeline_smoke.py
from miarag.ingestion import ReportDoc
from miarag.corpus import chunk_documents, split_members
from miarag.metrics import evaluate
from miarag.attacks.s2mia import s2mia_scores

class _RAG:
    def query(self, question, max_tokens=256):
        from miarag.rag import RAGResponse
        return RAGResponse(answer=question, retrieved_ids=["x"], perplexity=None)
    def perplexity_of(self, text): return 3.0

def test_pipeline_runs_end_to_end():
    docs = [ReportDoc(f"d{i}", f"C{i}", ("token%d " % i) * 200, has_person=False) for i in range(4)]
    members, non_members = split_members(chunk_documents(docs, 200, 20), 0.5, 42)
    scores, labels = s2mia_scores(_RAG(), members + non_members)
    rep = evaluate(scores, labels, prior=0.1)
    assert 0.0 <= rep.auc <= 1.0
    assert "ppv" in rep.to_row()
