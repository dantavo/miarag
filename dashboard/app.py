"""Streamlit dashboard di presentazione per il PoC MIA-on-RAG."""
import json
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd

from miarag.config import get_settings
from miarag.corpus import chunk_documents, split_members, ReportDoc
from miarag.rag import TargetRAG
from miarag.metrics import evaluate
from miarag.ingestion import ingest_file
from miarag.dashboard_helpers import (
    load_summary_rows, pii_demo, live_s2mia_on_chunk, live_budgetleak_on_chunk
)

# Header e disclaimer
st.set_page_config(page_title="MIA-on-RAG PoC Dashboard", layout="wide")
st.title("MIA-on-RAG PoC Dashboard")
st.caption(
    "PoC tesi su Membership Inference Attack in sistemi RAG. "
    "Documenti pseudonimizzati. Nessun dato reale/PII in output."
)

settings = get_settings()

def _load_reports(path: Path) -> list[ReportDoc]:
    """Carica reports.jsonl → lista di ReportDoc."""
    if not path.exists():
        return []
    docs = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            docs.append(ReportDoc(**d))
    return docs

@st.cache_resource
def _get_rag_with_corpus():
    """Costruisce RAG + corpus solo quando serve, con cache. Ritorna (rag, members, non_members) o None se manca reports.jsonl."""
    reports_path = settings.data_dir / "processed" / "reports.jsonl"
    if not reports_path.exists():
        return None
    docs = _load_reports(reports_path)
    if not docs:
        return None
    chunks = chunk_documents(docs)
    members, non_members = split_members(chunks, frac=0.5, seed=settings.seed)
    if not members:
        return None
    rag = TargetRAG(
        embedding_model=settings.embedding_model,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        top_k=settings.top_k
    )
    rag.index(members)
    return rag, members, non_members

# Sidebar per navigazione
st.sidebar.title("Navigazione")
section = st.sidebar.radio(
    "Sezione",
    ["Attacchi Live", "Metriche & Grafici", "Gate PII", "Esplora RAG", "Ingest Documenti"]
)

# ====== SEZIONE 1: ATTACCHI LIVE ======
if section == "Attacchi Live":
    st.header("Attacchi Live su Ollama")
    st.info("Esegue S2MIA o BudgetLeak su un sottoinsieme di chunk (lento, richiede Ollama attivo).")
    st.info("Nota: il RAG è indicizzato SOLO con i chunk membri (simulazione dell'attacco); i non-membri sono testati contro questo RAG per misurare se l'attacco riesce a distinguerli.")

    max_chunks = st.number_input("Max chunk da testare", min_value=1, max_value=20, value=5, step=1)
    attack_type = st.radio("Tipo attacco", ["S2MIA", "BudgetLeak"])

    if st.button(f"Esegui {attack_type}"):
        corpus_data = _get_rag_with_corpus()
        if corpus_data is None:
            st.error("reports.jsonl mancante o vuoto. Esegui prima l'ingestion.")
        else:
            rag, members, non_members = corpus_data
            # Prendi i primi max_chunks membri e non-membri
            test_members = members[:max_chunks]
            test_non_members = non_members[:max_chunks]

            results = []
            with st.spinner(f"Esecuzione {attack_type} su {len(test_members + test_non_members)} chunk..."):
                for chunk in test_members:
                    try:
                        if attack_type == "S2MIA":
                            res = live_s2mia_on_chunk(rag, chunk.text)
                        else:
                            res = live_budgetleak_on_chunk(rag, chunk.text)
                        results.append({
                            "chunk_id": chunk.chunk_id,
                            "is_member": True,
                            "score": res["score"]
                        })
                    except Exception as e:
                        st.error(f"Errore su chunk {chunk.chunk_id}: {e}")

                for chunk in test_non_members:
                    try:
                        if attack_type == "S2MIA":
                            res = live_s2mia_on_chunk(rag, chunk.text)
                        else:
                            res = live_budgetleak_on_chunk(rag, chunk.text)
                        results.append({
                            "chunk_id": chunk.chunk_id,
                            "is_member": False,
                            "score": res["score"]
                        })
                    except Exception as e:
                        st.error(f"Errore su chunk {chunk.chunk_id}: {e}")

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Valuta se abbiamo almeno un membro e un non-membro
                scores = [r["score"] for r in results]
                labels = [r["is_member"] for r in results]
                if any(labels) and not all(labels):
                    metrics = evaluate(scores, labels, prior=0.1)
                    st.success(f"AUC: {metrics['auc']:.3f} | TPR@1%FPR: {metrics['tpr_at_1fpr']:.3f}")
                else:
                    st.warning("Serve almeno un membro e un non-membro per calcolare AUC.")

# ====== SEZIONE 2: METRICHE & GRAFICI ======
elif section == "Metriche & Grafici":
    st.header("Metriche & Grafici")
    st.info("Visualizza risultati precedenti da results/.")

    summary_path = settings.results_dir / "summary.csv"
    rows = load_summary_rows(summary_path)

    if rows:
        st.subheader("Summary Metriche")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("Nessun summary.csv trovato. Esegui run_attack.py + run_eval.py.")

    roc_path = settings.results_dir / "roc.png"
    if roc_path.exists():
        st.subheader("ROC Curve")
        st.image(str(roc_path), use_column_width=True)
    else:
        st.info("Nessun roc.png trovato.")

# ====== SEZIONE 3: GATE PII ======
elif section == "Gate PII":
    st.header("Demo Gate PII")
    st.info("Mostra come il testo grezzo viene pseudonimizzato (solo regex per velocità UI; full-run usa anche NER).")

    user_text = st.text_area("Inserisci testo con PII", height=150, value="Amministratore CF RSSMRA80A01F205X presente. P.IVA 01234567890.")

    if st.button("Pseudonimizza"):
        if user_text.strip():
            pseudo = pii_demo(user_text)
            st.subheader("Testo pseudonimizzato")
            st.code(pseudo, language=None)
        else:
            st.warning("Inserisci del testo.")

# ====== SEZIONE 4: ESPLORA RAG ======
elif section == "Esplora RAG":
    st.header("Esplora RAG Black-Box")
    st.info("Query il RAG e vedi risposta + chunk recuperati.")

    query = st.text_input("Inserisci query")

    if st.button("Query RAG"):
        if not query.strip():
            st.warning("Inserisci una query.")
        else:
            corpus_data = _get_rag_with_corpus()
            if corpus_data is None:
                st.error("reports.jsonl mancante o Ollama non disponibile. Esegui prima l'ingestion.")
            else:
                rag, _, _ = corpus_data
                try:
                    with st.spinner("Interrogazione RAG..."):
                        resp = rag.query(query, max_tokens=256)
                    st.subheader("Risposta")
                    st.write(resp.answer)
                    st.subheader("Chunk recuperati")
                    st.code(", ".join(resp.retrieved_ids), language=None)
                except Exception as e:
                    st.error(f"Errore nel query RAG: {e}")

# ====== SEZIONE 5: INGEST DOCUMENTI ======
elif section == "Ingest Documenti":
    st.header("Ingest Nuovi Documenti")
    st.info("Carica PDF/DOCX/MD/TXT → pseudonimizza → aggiungi a reports.jsonl. NB: richiede reindicizzazione RAG per attacchi.")

    uploaded_file = st.file_uploader("Carica documento", type=["pdf", "docx", "md", "txt"])

    if uploaded_file is not None:
        if st.button("Ingest"):
            with st.spinner("Ingestion in corso (con NER reale)..."):
                try:
                    # Salva upload in file temporaneo
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = Path(tmp.name)

                    out_path = settings.data_dir / "processed" / "reports.jsonl"
                    # NER reale di produzione, append=True
                    doc = ingest_file(tmp_path, out_path, ner=None, append=True)

                    # Rimuovi il file temporaneo
                    tmp_path.unlink()

                    # Mostra solo metadati, MAI il testo (già pseudonimizzato ma non stampabile per policy)
                    st.success("Ingestion completata!")
                    st.write(f"**Company:** {doc.company}")
                    st.write(f"**Has Person:** {doc.has_person}")
                    st.write(f"**Lunghezza testo:** {len(doc.text)} caratteri")
                    st.info("Per usare questo documento negli attacchi, riavvia la dashboard (cache RAG).")
                except Exception as e:
                    st.error(f"Errore ingestion: {e}")
