# src/miarag/ingestion.py
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from pypdf import PdfReader
from miarag.pseudonymize import PII_PATTERNS, pseudonymize_text, NerDetector, RizzoNerDetector

@dataclass
class ReportDoc:
    doc_id: str
    company: str
    text: str
    has_person: bool

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def parse_report_text(raw: str, doc_id: str, ner: NerDetector | None = None) -> ReportDoc:
    # Derive has_person from NER FULLNAME detections on raw text
    has_person = False
    if ner is not None:
        fullname_spans = [s for s in ner.detect(raw) if s[2] == "PERSON"]
        has_person = len(fullname_spans) > 0

    company = next((ln.strip() for ln in raw.splitlines() if ln.strip()), doc_id)  # dal raw
    # Normalizza spazi unicode (nbsp \xa0 e affini): senza questo i numeri delle
    # tabelle di bilancio restano spezzati e sfuggono a regex/NER → leak PII.
    clean = raw.replace("\xa0", " ").replace(" ", " ").replace(" ", " ")
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    clean = pseudonymize_text(clean, ner=ner)
    return ReportDoc(doc_id=doc_id, company=company, text=clean, has_person=has_person)

def ingest_dir(corpus_dir: Path, out_path: Path, ner: NerDetector | None = None) -> list[ReportDoc]:
    if ner is None:
        ner = RizzoNerDetector()          # produzione: NER italiano attivo
    docs = []
    for pdf in sorted(corpus_dir.glob("*.pdf")):
        raw = load_pdf(pdf)
        docs.append(parse_report_text(raw, doc_id=pdf.stem, ner=ner))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    return docs
