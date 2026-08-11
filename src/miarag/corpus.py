# src/miarag/corpus.py
import random
from dataclasses import dataclass, replace
from miarag.ingestion import ReportDoc

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    is_member: bool
    has_person: bool

def chunk_documents(docs: list[ReportDoc], chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    for d in docs:
        t = d.text
        idx = 0
        for start in range(0, max(1, len(t)), step):
            piece = t[start:start + chunk_size]
            if not piece.strip():
                continue
            chunks.append(Chunk(
                chunk_id=f"{d.doc_id}::{idx}", doc_id=d.doc_id,
                text=piece, is_member=False, has_person=d.has_person))
            idx += 1
    return chunks

def split_members(chunks: list[Chunk], member_frac: float = 0.5, seed: int = 42):
    rng = random.Random(seed)
    shuffled = chunks[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * member_frac)
    members = [replace(c, is_member=True) for c in shuffled[:cut]]
    non_members = [replace(c, is_member=False) for c in shuffled[cut:]]
    return members, non_members
