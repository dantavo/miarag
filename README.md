# MIA-RAG PoC — Membership Inference Attacks su un sistema RAG

Proof-of-Concept sperimentale a corredo della tesi magistrale: verifica se un
attaccante *black-box* può stabilire quali documenti fanno parte del corpus
indicizzato da un sistema **RAG** (Retrieval-Augmented Generation), interrogando
il sistema solo attraverso la sua interfaccia (domanda → risposta).

> ⚠️ **Privacy / dati.** Il corpus reale (report aziendali con PII) **non è nel
> repository** ed è escluso via `.gitignore` (`documenti/`, `data/`, `results/`,
> `.env`). Prima di essere indicizzato, ogni documento passa da una pipeline di
> **pseudonimizzazione** che sostituisce gli identificatori diretti (nomi, CF,
> P.IVA, IBAN, email, telefoni, indirizzi, …) con token deterministici
> `<TIPO>_<hash8>`, con un *backstop fail-closed* su qualunque sequenza ≥9 cifre
> residua. Il repository contiene **solo codice e test sintetici**.

## Cosa misura

Due attacchi implementati sopra la stessa interfaccia black-box, più le baseline
descritte in tesi:

- **S2MIA** — segnale da *BLEU* (sovrapposizione risposta↔documento) + *perplexity*
  (proxy LM locale).
- **BudgetLeak** — side-channel sul *generation budget* (`num_predict` di Ollama).

Metriche di attacco (modulo `metrics`): **AUC/ROC**, **TPR@1%FPR**, **PPV con prior**
di membership (forma Bayesiana `PPV = π·TPR / (π·TPR + (1−π)·FPR)`), **membership
advantage** (`TPR − FPR`).

## Stack

Python 3.11, ambiente `uv`, tutto **locale** su Apple Silicon (accelerazione MPS/Metal).

- **RAG:** LangChain + **Chroma** (vector DB in-memory) + `sentence-transformers`
  (MiniLM, embedding locali).
- **Target LLM:** **Ollama** (`llama3.1:8b`) come modello primario. Opzioni
  intercambiabili previste: GPT-4o-mini via Azure OpenAI, Claude via AWS Bedrock
  (chiavi in `.env`).
- **Perplexity:** Ollama v0.32.7 non espone logprobs via `/api/generate` → proxy
  GPT-2 locale via `transformers` (`exp(mean NLL)`).
- **PII:** regex (CF/P.IVA/REA) + NER italiano `rizzo-pii-0.3B` (`transformers`).

## Struttura

```
src/miarag/
  config.py         # Settings (frozen dataclass) + get_settings()
  pseudonymize.py   # gate PII: regex + NER → token deterministici, backstop ≥9 cifre
  ingestion.py      # PDF → testo → normalizzazione unicode → pseudonimizzazione
  corpus.py         # chunking (finestra char, overlap) + split membri/non-membri
  rag.py            # TargetRAG: interfaccia black-box (Chroma + Ollama) + perplexity
  metrics.py        # AUC, TPR@FPR, PPV-con-prior, membership advantage
tests/              # suite offline (network-free), NER finto per i test
scripts/
  run_ingestion.py  # ingest reale del corpus (fuori dal repo)
```

## Setup

```bash
# dipendenze
uv sync

# modello target locale (pull ~5 GB, su WiFi stabile)
ollama pull llama3.1:8b

# chiavi API (solo se si usano i target commerciali) — mai committare .env
cp .env.example .env   # poi compilare
```

## Test

Suite completamente offline (nessuna rete, NER e Ollama mockati):

```bash
uv run pytest -q
```

## Ingestion del corpus reale

Il corpus vive **fuori dal repo** (in `documenti/`, git-ignored). L'ingest
pseudonimizza e scrive JSONL in `data/processed/`:

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=src uv run python scripts/run_ingestion.py
```

## Stato

| Componente | Stato |
|---|---|
| Config + scaffold | ✅ |
| Pseudonimizzazione PII (gate etico) | ✅ — 0 PII residua su ingest reale (5 PDF) |
| Ingestion PDF | ✅ |
| Chunking + split membri | ✅ |
| TargetRAG (black-box) | ✅ |
| Metriche di attacco | ✅ |
| Attacco S2MIA | ⏳ |
| Attacco BudgetLeak | ⏳ |
| Plot / grafici | ⏳ |
| Orchestrazione end-to-end | ⏳ |
| Difese + trade-off | ⏳ |

## Note

- Documentazione operativa: [`OLLAMA.md`](OLLAMA.md) (setup modello locale),
  [`DS4_EC2.md`](DS4_EC2.md) (infrastruttura remota opzionale).
- Il versioning del codice PoC è separato dal Vault della tesi.
