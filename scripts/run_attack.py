# scripts/run_attack.py
import csv, json
from pathlib import Path
from miarag.config import get_settings
from miarag.ingestion import ReportDoc
from miarag.corpus import chunk_documents, split_members, Chunk
from miarag.rag import TargetRAG
from miarag.attacks.s2mia import s2mia_scores
from miarag.attacks.budgetleak import budgetleak_scores

def _load_reports(path: Path) -> list[ReportDoc]:
    docs = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        docs.append(ReportDoc(**d))
    return docs

def build_pipeline(reports_jsonl: Path, rag) -> tuple[list[Chunk], list[Chunk]]:
    docs = _load_reports(reports_jsonl)
    members, non_members = split_members(chunk_documents(docs), 0.5, 42)
    rag.index(members)          # SOLO i membri sono indicizzati
    return members, non_members

def _save(path: Path, chunks, scores, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["chunk_id", "score", "label", "has_person"])
        for c, s, l in zip(chunks, scores, labels):
            w.writerow([c.chunk_id, s, l, int(c.has_person)])

def main():
    s = get_settings()
    rag = TargetRAG(s.embedding_model, s.ollama_base_url, s.ollama_model, s.top_k)
    members, non_members = build_pipeline(s.data_dir / "processed" / "reports.jsonl", rag)
    targets = members + non_members
    for name, fn in [("s2mia", s2mia_scores), ("budgetleak", budgetleak_scores)]:
        scores, labels = fn(rag, targets)
        _save(s.results_dir / f"scores_{name}.csv", targets, scores, labels)
        print(f"{name}: saved {len(scores)} scores")

if __name__ == "__main__":
    main()
