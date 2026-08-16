# src/miarag/ingestion.py
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from miarag.pseudonymize import PII_PATTERNS, pseudonymize_text, NerDetector, ItalianPIINerDetector

@dataclass
class ReportDoc:
    doc_id: str
    company: str
    text: str
    has_person: bool

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def load_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def load_md(path: Path) -> str:
    # markdown trattato come testo grezzo: la pseudonimizzazione lavora sul testo,
    # la sintassi md non è rilevante per il corpus. Nessuna dipendenza extra.
    return path.read_text(encoding="utf-8", errors="replace")

_LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".md": load_md, ".txt": load_txt}

def load_any(path: Path) -> str:
    fn = _LOADERS.get(path.suffix.lower())
    if fn is None:
        raise ValueError(f"formato non supportato: {path.suffix} (attesi: {sorted(_LOADERS)})")
    return fn(path)

def ingest_file(path: Path, out_path: Path, ner: NerDetector | None = None, append: bool = True) -> ReportDoc:
    """Ingerisce UN file (pdf/docx/md/txt): carica, pseudonimizza, appende un ReportDoc a out_path (JSONL).
    ner=None → ItalianPIINerDetector reale (produzione). append=False sovrascrive out_path."""
    if ner is None:
        ner = ItalianPIINerDetector()
    raw = load_any(path)
    doc = parse_report_text(raw, doc_id=path.stem, ner=ner)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    return doc

def parse_report_text(raw: str, doc_id: str, ner: NerDetector | None = None) -> ReportDoc:
    # Derive has_person from NER FULLNAME detections on raw text
    has_person = False
    if ner is not None:
        fullname_spans = [s for s in ner.detect(raw) if s[2] == "PERSON"]
        has_person = len(fullname_spans) > 0

    company = next((ln.strip() for ln in raw.splitlines() if ln.strip()), doc_id)  # dal raw
    # Normalizza spazi unicode (nbsp \xa0 e affini): senza questo i numeri delle
    # tabelle di bilancio restano spezzati e sfuggono a regex/NER → leak PII.
    clean = raw.replace("\xa0", " ").replace(" ", " ").replace(" ", " ")
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    clean = pseudonymize_text(clean, ner=ner)
    # Sanitize doc_id and company too: they may carry organization names (e.g.
    # from the source filename or the first document line) that would otherwise
    # leak into chunk_ids (chunk_id = f"{doc_id}::{idx}") and scores CSVs.
    # Regex-only (no NER) is enough here and keeps it fast.
    doc_id = pseudonymize_text(doc_id, ner=None)
    company = pseudonymize_text(company, ner=ner)
    return ReportDoc(doc_id=doc_id, company=company, text=clean, has_person=has_person)

def ingest_dir(corpus_dir: Path, out_path: Path, ner: NerDetector | None = None) -> list[ReportDoc]:
    if ner is None:
        ner = ItalianPIINerDetector()          # produzione: NER italiano attivo
    docs = []
    files = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() in _LOADERS)
    for src in files:
        raw = load_any(src)
        docs.append(parse_report_text(raw, doc_id=src.stem, ner=ner))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    return docs
