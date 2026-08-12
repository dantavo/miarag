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
    members, non_members = split_members(chunks, member_frac=0.5, seed=settings.seed)
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
    st.markdown("""
    **Esegue attacchi di Membership Inference in tempo reale** contro il RAG locale (Ollama).

    - **S2MIA**: misura sovrapposizione risposta↔documento (BLEU) + perplexity. Score alto → probabile membro.
    - **BudgetLeak**: side-channel sul generation budget (32/96/256 token). Sequenze di similarità crescenti → membro.

    **Setup attacco**: il RAG è indicizzato SOLO con i chunk membri (split 50/50 del corpus). I non-membri sono testati *contro* questo RAG indicizzato — se l'attacco funziona, score(membri) > score(non-membri) → AUC > 0.5 (casuale).

    ⚠️ **Lento**: ogni chunk richiede 1-3 generazioni Ollama (~5-10s cadauno). Limita a 5-10 chunk per test rapido.
    """)
    st.info("💡 **Per la tesi**: questi attacchi dimostrano che un attaccante black-box (solo query→risposta) può inferire membership del corpus RAG senza accesso ai pesi del modello.")

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
            all_chunks = [(c, True) for c in test_members] + [(c, False) for c in test_non_members]
            total = len(all_chunks)

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (chunk, is_member) in enumerate(all_chunks):
                status_text.text(f"Attacco {attack_type}: chunk {i+1}/{total} ({'membro' if is_member else 'non-membro'})...")
                try:
                    if attack_type == "S2MIA":
                        res = live_s2mia_on_chunk(rag, chunk.text)
                    else:
                        res = live_budgetleak_on_chunk(rag, chunk.text)
                    results.append({
                        "chunk_id": chunk.chunk_id,
                        "is_member": is_member,
                        "score": res["score"]
                    })
                except Exception as e:
                    st.error(f"Errore su chunk {chunk.chunk_id}: {e}")

                progress_bar.progress((i + 1) / total)

            status_text.text(f"✓ Completato: {total} chunk processati")
            progress_bar.empty()

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Valuta se abbiamo almeno un membro e un non-membro
                scores = [r["score"] for r in results]
                labels = [r["is_member"] for r in results]

                # Guard NaN/inf (GPT-2 perplexity fail, embedding error)
                import math
                valid_idx = [i for i, s in enumerate(scores) if math.isfinite(s)]
                if len(valid_idx) < len(scores):
                    st.warning(f"⚠️ {len(scores) - len(valid_idx)} chunk con score NaN/inf ignorati (errore perplexity/embedding)")
                scores = [scores[i] for i in valid_idx]
                labels = [labels[i] for i in valid_idx]

                if len(scores) >= 2 and any(labels) and not all(labels):
                    metrics = evaluate(scores, labels, prior=0.1)
                    st.success(f"AUC: {metrics.auc:.3f} | TPR@1%FPR: {metrics.tpr_at_1fpr:.3f}")
                elif len(scores) < 2:
                    st.warning("Troppi chunk con errore, impossibile calcolare metriche.")
                else:
                    st.warning("Serve almeno un membro e un non-membro per calcolare AUC.")

# ====== SEZIONE 2: METRICHE & GRAFICI ======
elif section == "Metriche & Grafici":
    st.header("Metriche & Grafici")
    st.markdown("""
    **Visualizza i risultati del full-run** (già eseguito su tutti i 245 chunk del corpus).

    **Metriche chiave** (prior membership = 0.1):
    - **AUC**: area sotto curva ROC. >0.5 = meglio di casuale; ≥0.7 = attacco forte.
    - **TPR@1%FPR**: true positive rate quando FPR ≤ 1% (setting conservativo: attaccante accetta max 1% falsi positivi).
    - **PPV (Positive Predictive Value)**: probabilità che un "membro predetto" sia davvero membro, corretto per prior π=0.1 (formula Bayesiana: π·TPR / (π·TPR + (1−π)·FPR)).
    - **Advantage**: TPR − FPR (guadagno netto sopra random guess).

    **Risultati tesi** (defense=none, embedding MiniLM-L6-v2, 5 documenti corpus):
    - S2MIA: AUC 0.527 — lievemente sopra casuale, corpus piccolo limita segnale.
    - BudgetLeak: AUC 0.521 — simile, side-channel budget debole su corpus ridotto.

    ℹ️ Con corpus 20-50 doc, AUC attesa 0.6-0.75+ (vedi letteratura Carlini et al. 2021).
    """)

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
        st.image(str(roc_path), use_container_width=True)
    else:
        st.info("Nessun roc.png trovato.")

# ====== SEZIONE 3: GATE PII ======
elif section == "Gate PII":
    st.header("Demo Gate PII")
    st.markdown("""
    **Pseudonimizzazione etica** del corpus prima dell'indicizzazione RAG.

    **Pipeline full** (ingestion reale):
    1. **Regex** → CF, P.IVA, IBAN, email, telefono, indirizzi, numeri lunghi (≥9 cifre).
    2. **NER italiano** (`rizzo-pii-0.3B`) → nomi persona, luoghi, org, date, numeri ID.
    3. **Backstop fail-closed**: qualunque sequenza ≥9 cifre residua → token `NUM_<hash8>`.

    Ogni entità PII → token deterministico `<TIPO>_<hash8>` (stesso input = stesso token, per preservare co-occorrenze senza esporre PII).

    **Qui (demo UI)**: solo regex (veloce), no NER. Il full-run con NER ha verificato **0 PII residua** su 5 PDF reali (507 token pseudonimizzati, 0 email/IBAN/CF/9+cifre nel testo finale).

    🔒 **Ethical clearance**: nessun dato PII reale entra nel RAG né nel repository GitHub. Artefatti (reports.jsonl, results/) git-ignored.
    """)

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
    st.markdown("""
    **Interroga il sistema RAG** come farebbe un attaccante o un utente normale.

    **Architettura RAG** (locale):
    - **Vector DB**: Chroma (in-memory, ephemeral).
    - **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (384d, generico non-IT).
    - **Retriever**: top-k=4 chunk più simili (cosine similarity).
    - **Generator**: Ollama `llama3.1:8b` (quantizzato Q4_K_M).

    **Interfaccia black-box**: l'attaccante vede solo query → risposta (+ opzionalmente gli ID chunk recuperati, se l'applicazione li espone). NON vede embedding, pesi modello, o contenuto pieno dei chunk — solo la risposta generata.

    Gli attacchi MIA sfruttano *pattern nella risposta* (BLEU, perplexity, lunghezza generata) per inferire se un chunk era nel corpus indicizzato.
    """)

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
    st.markdown("""
    **Aggiungi documenti al corpus** (formato PDF, DOCX, Markdown, TXT).

    **Pipeline**:
    1. **Carica** file da UI.
    2. **Estrai testo grezzo** (pypdf, python-docx, stdlib).
    3. **Normalizza** (unicode space → ASCII, rimuovi nbsp).
    4. **Pseudonimizza** (regex + NER `rizzo-pii-0.3B` → token `<TIPO>_<hash8>` + backstop ≥9 cifre).
    5. **Append** a `data/processed/reports.jsonl` (4 campi: doc_id, company, text, has_person).

    **NB**: dopo l'ingest, il RAG deve essere **reindicizzato** (riavvia dashboard) perché la cache `@st.cache_resource` tiene il vecchio corpus. Per test produttivi, rigenera il full-run con `run_ingestion.py` (processa l'intero `documenti/` batch).

    📄 **Formati supportati**: PDF (via pypdf), DOCX (via python-docx), Markdown, plain text. Ogni formato passa per la stessa pipeline PII (ethical gate obbligatorio).
    """)

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
