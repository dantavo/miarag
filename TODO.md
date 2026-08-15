# TODO / Roadmap post-refactor v0.2

State: everything merged into `main`. Refactor branch and `v0.1-thesis` tag
have been cleaned up during history rewrite.

## Done

- [x] **Pluggable providers**: Protocol for LLM / Embedding / Perplexity
  (`src/miarag/providers/`).
- [x] Backends: `ollama`, `azure_openai`, `bedrock`, `sentence_tf`,
  `openai_embed`, `gpt2`, `hf_causal` (generic, useful for Italian LMs).
- [x] Registry + factory with lazy import (no heavy boot cost).
- [x] `Settings.validate()` fail-fast on missing env vars.
- [x] `TargetRAG` with DI (`llm=`, `embedder=`, `ppl=`) + backcompat for the
  positional v0.1-thesis signature.
- [x] CLI `run_attack.py`: flags `--llm/--embed/--ppl`.
- [x] Dashboard: provider selectbox in the sidebar.
- [x] BudgetLeak: `FINAL_SIM_IDX` constant (no more magic index `cntr[:, 2]`).
- [x] S2MIA: split on punctuation (token-mid fallback), optional cosine
  similarity feature via `use_cosine=True`.
- [x] Retry/backoff with `tenacity` for paid API providers (Azure, Bedrock).
- [x] Token/cost tracker (`providers/_cost.py`, `TRACKER` singleton).
- [x] Optional persistent Chroma via `persist_dir=...` in TargetRAG.
- [x] Provider-agnostic tests (`tests/test_providers.py`): Protocol contract,
  DI composition, swap provider, cost tracking, persistent chroma.
- [x] **RAG-MIA (Anderson 2025)** as third attack: prompt injection black-box.
  `src/miarag/attacks/rag_mia.py` + 13 tests. Wired into `run_attack.py`,
  `run_eval.py`, dashboard live attacks. Score in [0,1]: Yes=1, No=0,
  refusal=0.5 (defense-aware for prompt hardening).
  **76 total tests, 75 pass, 1 known skip**.

## To do (decreasing priority)

### High

- [ ] **Italian PPL as default**: switch to `PERPLEXITY_PROVIDER=hf_causal` +
  `PERPLEXITY_HF_MODEL=sapienzanlp/Minerva-350M-base-v1.0` as the default in
  `.env.example` and document in README (measurable impact on S2MIA AUC for
  IT corpora vs GPT-2 EN). Currently gated on HF approval — evaluate
  non-gated alternatives.
- [ ] **macOS ARM segfault** on the full suite when
  `test_backcompat_positional_signature` runs together with xgboost+tqdm:
  currently skipped. Root cause: tqdm monitor thread + torch cleanup +
  xgboost thread pool. Try `TQDM_DISABLE=1` or disable xgboost threads
  (`OMP_NUM_THREADS=1`).
- [ ] Wire `use_cosine=True` as a CLI flag in `run_attack.py` for S2MIA-M.

### Medium

- [ ] **Rate limit / hard cost cap** on TRACKER: `TRACKER.max_calls`, raise
  above threshold.
- [ ] **Native logprobs** on Azure OpenAI (`supports_logprobs=True`):
  implement `AzureLogprobPerplexity` using `logprobs=True` instead of the HF
  proxy → S2MIA signal more faithful to the target model.
- [ ] Persist Chroma by default in `data/chroma/` for `run_attack.py` (avoids
  reindexing on every run for large corpora).
- [ ] BERTScore as an additional S2MIA feature (on top of BLEU + cosine).

### Low / nice-to-have

- [ ] `gemini` provider (Google Vertex AI) as a 4th option.
- [ ] `local_hf` provider (HuggingFace transformers) as a target LLM for
  comparison against Ollama on the same model (Llama-3.1 8B in HF format).
- [ ] Dashboard: live cost tracker panel (`TRACKER.snapshot()`).
- [ ] Migrate `dataclass(frozen=True)` → `pydantic.BaseModel` for Settings
  (type validation + serialization).
- [ ] CI on GitHub Actions with matrix (Python 3.11/3.12, macOS/Linux).
