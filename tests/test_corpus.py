# tests/test_corpus.py
from miarag.ingestion import ReportDoc
from miarag.corpus import chunk_documents, split_members, split_members_by_doc, Chunk

def _docs():
    return [ReportDoc(f"d{i}", f"C{i}", ("parola " * 400).strip(), has_person=bool(i % 2))
            for i in range(4)]

def test_chunking_produces_chunks():
    chunks = chunk_documents(_docs(), chunk_size=200, overlap=20)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert len(chunks) > len(_docs())          # ogni doc si spezza
    assert all(len(c.text) <= 200 for c in chunks)

def test_split_is_deterministic_and_disjoint():
    chunks = chunk_documents(_docs(), chunk_size=200, overlap=20)
    m1, n1 = split_members(chunks, member_frac=0.5, seed=42)
    m2, n2 = split_members(chunks, member_frac=0.5, seed=42)
    ids_m1 = {c.chunk_id for c in m1}
    assert ids_m1 == {c.chunk_id for c in m2}   # deterministico
    assert ids_m1.isdisjoint({c.chunk_id for c in n1})  # disgiunti
    assert all(c.is_member for c in m1)
    assert all(not c.is_member for c in n1)

def test_split_by_doc_no_doc_crosses_sides():
    """Document-level split: nessun documento ha chunk su entrambi i lati.
    Questo elimina la fuga dei chunk-fratelli (causa del basso AUC)."""
    chunks = chunk_documents(_docs(), chunk_size=200, overlap=20)
    members, non_members = split_members_by_doc(chunks, member_frac=0.5, seed=42)
    mem_docs = {c.doc_id for c in members}
    non_docs = {c.doc_id for c in non_members}
    # Un documento sta tutto da una parte: gli insiemi di doc_id sono disgiunti.
    assert mem_docs.isdisjoint(non_docs)
    assert all(c.is_member for c in members)
    assert all(not c.is_member for c in non_members)

def test_split_by_doc_deterministic_and_balanced():
    chunks = chunk_documents(_docs(), chunk_size=200, overlap=20)
    m1, _ = split_members_by_doc(chunks, member_frac=0.5, seed=42)
    m2, _ = split_members_by_doc(chunks, member_frac=0.5, seed=42)
    assert {c.chunk_id for c in m1} == {c.chunk_id for c in m2}   # deterministico
    # Bilanciamento ragionevole: membri tra 30% e 70% del totale.
    assert 0.3 * len(chunks) <= len(m1) <= 0.7 * len(chunks)
