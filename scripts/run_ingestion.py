# scripts/run_ingestion.py
from miarag.config import get_settings
from miarag.ingestion import ingest_dir

def main():
    s = get_settings()
    out = s.data_dir / "processed" / "reports.jsonl"
    docs = ingest_dir(s.corpus_dir, out)
    print(f"Ingested {len(docs)} reports -> {out}")
    for d in docs:
        print(f"  {d.doc_id}: {d.company} (has_person={d.has_person})")

if __name__ == "__main__":
    main()
