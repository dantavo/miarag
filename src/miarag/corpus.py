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


def split_members_by_doc(chunks: list[Chunk], member_frac: float = 0.5, seed: int = 42):
    """Document-level membership split (literature-standard, Li 2025 / Anderson 2025).

    Whole documents are assigned to member (indexed) or non-member (held-out, never
    indexed). This removes the leakage of the chunk-level split, where a non-member
    chunk's sibling chunks from the SAME document are indexed and get retrieved,
    making members and non-members near-indistinguishable.

    Balancing: documents have very uneven chunk counts, so a naive 50/50 doc split
    can badly unbalance chunk counts. We greedily assign documents (largest first)
    to whichever side is currently furthest below its chunk target → keeps the
    member/non-member CHUNK counts close to `member_frac`.
    """
    # Group chunk indices by document.
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    total = len(chunks)
    target_member = total * member_frac

    # Shuffle doc order deterministically, then sort by size desc for greedy balance.
    doc_ids = list(by_doc.keys())
    random.Random(seed).shuffle(doc_ids)
    doc_ids.sort(key=lambda d: len(by_doc[d]), reverse=True)

    member_docs: set[str] = set()
    member_count = 0
    for d in doc_ids:
        n = len(by_doc[d])
        # Assign to member side if it keeps us at/under target; else non-member.
        if member_count + n <= target_member or member_count == 0:
            member_docs.add(d)
            member_count += n

    members, non_members = [], []
    for c in chunks:
        if c.doc_id in member_docs:
            members.append(replace(c, is_member=True))
        else:
            non_members.append(replace(c, is_member=False))
    return members, non_members
