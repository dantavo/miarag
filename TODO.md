# TODO / Roadmap

PoC **completa**. Questo file traccia cosa è stato fatto e i possibili sviluppi
futuri (utile per il capitolo "lavori futuri" della tesi). Nessun item residuo è
bloccante.

## Fatto

### Framework
- [x] Provider pluggable (Protocol LLM/Embedding/Perplexity): `ollama`,
  `azure_openai`, `bedrock`, `sentence_tf`, `openai_embed`, `gpt2`, `hf_causal`.
- [x] Registry + factory (lazy import), `Settings.validate()` fail-fast,
  `get_settings()` legge env a runtime (flag CLI `--llm/--embed/--ppl` effettivi).
- [x] `TargetRAG` con DI + backcompat firma v0.1-thesis.
- [x] Split a documento (`--split doc`) oltre al chunk-level legacy.
- [x] Retry/backoff + **gestione 429 (Retry-After) + throttling** (Azure).
- [x] Resilienza per-chunk (un blip salta 1 chunk, non uccide l'attacco).
- [x] Progress-log con ETA reale (`attacks/_common.scored_loop`).
- [x] Cost tracker (`providers/_cost.py`), Chroma persistente opzionale.

### Attacchi
- [x] **S2MIA** (BLEU + perplexity, split su punteggiatura, cosine opzionale).
- [x] **BudgetLeak** (Tri-Budget + Jaccard, FCM zero-knowledge).
- [x] **RAG-MIA** (Anderson 2025) prompt injection black-box.
- [x] **Gray-box via logprob** (Azure): RAG-MIA continuo `P(Sì)` (risolve
  TPR@1%FPR=0) + S2MIA perplexity nativa (no proxy GPT-2). Difesa applicata
  anche al path gray-box.

### Gate PII / etica
- [x] Regex IT-legal (CF/PIVA/REA/catasto) + NER italiano + nomi-azienda (env,
  fail-closed) + backstop cifre ≥9 + **redazione credenziali inline** (`SECRET_`).
- [x] Verifica fail-closed: 0 PII residua su corpus 44-doc.

### Esperimenti / output
- [x] Run 44-doc (33 wiki + 11 società), doc-split, 2 target (Azure/Ollama).
- [x] 6 grafici ROC/distribuzioni (`results/plots/`).
- [x] Dashboard Streamlit aggiornata (metriche 44-doc + galleria plot; avvio diretto).
- [x] Suite offline: **93 test, 1 skip noto**.

## Sviluppi futuri (opzionali — per "lavori futuri" in tesi)

### Sperimentali
- [ ] Difese complete su 44-doc (paraphrase + prompt_hardening su tutti gli
  attacchi/target). Interrotte da cadute di rete; lanciare sotto `caffeinate -dimsu`.
- [ ] BudgetLeak-P completa (attention-LSTM su shadow RAG) e multi-metrica
  (cosine/ROUGE/edit-distance su 14 budget) → robustezza reale alle difese
  (la variante ridotta Tri-Budget/Jaccard è fragile).
- [ ] PPL italiana di default (`hf_causal` + Minerva) — attualmente gated su HF.
- [ ] BERTScore come feature S2MIA aggiuntiva.

### Ingegneria
- [ ] Rate/cost cap hard su `TRACKER` (`max_calls`).
- [ ] Provider `gemini` (Vertex AI) e `local_hf` (Llama-3.1 in HF format).
- [ ] Fix segfault macOS ARM full-suite (torch+xgboost+tqdm): `TQDM_DISABLE=1`
  o `OMP_NUM_THREADS=1` (test `test_backcompat_positional_signature` skippato).
- [ ] Migrazione Settings a `pydantic.BaseModel`; CI GitHub Actions (matrix).
