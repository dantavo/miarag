# TODO / Roadmap post-refactor v0.2

Stato: `refactor/pluggable-providers` branch. Tag `v0.1-thesis` congela stato tesi.

## Fatto (commit `219701f` + successivi)

- [x] **Pluggable providers**: Protocol per LLM / Embedding / Perplexity
  (`src/miarag/providers/`).
- [x] Backend: `ollama`, `azure_openai`, `bedrock`, `sentence_tf`, `openai_embed`,
  `gpt2`, `hf_causal` (generico per LM italiani).
- [x] Registry + factory con lazy import (no cost boot pesante).
- [x] `Settings.validate()` fail-fast su env vars mancanti.
- [x] `TargetRAG` con DI (`llm=`, `embedder=`, `ppl=`) + backcompat firma
  positional v0.1-thesis.
- [x] CLI `run_attack.py`: flag `--llm/--embed/--ppl`.
- [x] Dashboard: selectbox provider in sidebar.
- [x] BudgetLeak: `FINAL_SIM_IDX` costante (no più magic index `cntr[:, 2]`).
- [x] S2MIA: split su punteggiatura (fallback token-mid), feature cosine
  opzionale via `use_cosine=True`.
- [x] Retry/backoff con `tenacity` per provider paid API (Azure, Bedrock).
- [x] Token/cost tracker (`providers/_cost.py`, `TRACKER` singleton).
- [x] Chroma persistente opzionale via `persist_dir=...` in TargetRAG.
- [x] Test provider-agnostic (`tests/test_providers.py`): Protocol contract,
  DI composition, swap provider, cost tracking, persistent chroma. **63 test
  totali, 62 passano, 1 skip noto**.

## Da fare (priorità decrescente)

### High

- [ ] **PPL italiano default**: cambiare `PERPLEXITY_PROVIDER=hf_causal` +
  `PERPLEXITY_HF_MODEL=sapienzanlp/Minerva-350M-base-v1.0` come default in
  `.env.example` e documentare in README (impatto misurabile su AUC S2MIA per
  corpus IT vs GPT-2 EN).
- [ ] **Segfault macOS ARM** su full suite quando `test_backcompat_positional_signature`
  gira insieme a xgboost+tqdm: attualmente skippato. Root cause: tqdm monitor
  thread + torch cleanup + xgboost thread pool. Provare `TQDM_DISABLE=1` o
  disabilitare xgboost threads (`OMP_NUM_THREADS=1`).
- [ ] Integrare `use_cosine=True` come flag CLI in `run_attack.py` per S2MIA-M.

### Medium

- [ ] **Rate limit / cost cap** hard su TRACKER: `TRACKER.max_calls`, raise oltre.
- [ ] **Logprobs nativi** su Azure OpenAI (`supports_logprobs=True`): implementare
  `AzureLogprobPerplexity` che usa `logprobs=True` invece di HF proxy → segnale
  S2MIA più fedele al modello target.
- [ ] Persist Chroma di default in `data/chroma/` per `run_attack.py`
  (evita reindex ogni run su corpus grande).
- [ ] BERTScore come feature aggiuntiva S2MIA (oltre BLEU + cosine).

### Low / nice-to-have

- [ ] Provider `gemini` (Google Vertex AI) come 4° opzione.
- [ ] Provider `local_hf` (HuggingFace transformers) come target LLM per confronto
  con Ollama sullo stesso modello (Llama-3.1 8B in HF format).
- [ ] Dashboard: pannello cost tracker live (TRACKER.snapshot()).
- [ ] Migrazione da `dataclass(frozen=True)` a `pydantic.BaseModel` per
  Settings (validazione tipi + serializzazione).
- [ ] CI GitHub Actions con matrix (Python 3.11/3.12, macOS/Linux).
