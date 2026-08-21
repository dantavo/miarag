# MIA-RAG PoC — Membership Inference Attacks on a RAG System

Experimental proof-of-concept accompanying a master's thesis: verify whether a
*black-box* attacker can determine which documents belong to the corpus
indexed by a **RAG** (Retrieval-Augmented Generation) system by querying it
only through its interface (question → answer).

> ⚠️ **Privacy / data.** The real corpus (business documents with PII) is **not**
> in this repository and is excluded via `.gitignore` (`documenti/`, `data/`,
> `results/`, `.env`). Before indexing, every document is processed by a
> **pseudonymization** pipeline that replaces direct identifiers (names, tax
> codes, VAT numbers, IBAN, email, phone, addresses, organization names, inline
> system credentials, …) with deterministic tokens `<TYPE>_<hash8>`, backed by a
> *fail-closed backstop* on any residual digit sequence ≥ 9 characters. The
> repository contains **only code and aggregate metrics**.

## TL;DR — results

Realistic scenario: an enterprise RAG over an internal technical wiki, into which
**client documents (with PII) are erroneously indexed**. Can a black-box attacker
detect them? Corpus: 44 documents → 1407 chunks, document-level split, targets
**GPT-4o-mini** (Azure) and **llama3.1:8b** (Ollama).

| Attack | GPT-4o-mini | llama3.1 | metric |
|---|---|---|---|
| S2MIA | 0.687 | 0.688 | AUC |
| BudgetLeak | 0.592 | 0.560 | AUC |
| RAG-MIA (black-box) | 0.796 | 0.769 | AUC |
| **RAG-MIA (gray-box, logprobs)** | **0.988** (TPR@1%FPR **0.750**) | — | AUC |

**Headline**: exposing token **logprobs** (available on the commercial API, not on
the local model) turns RAG-MIA near-perfect and defeats the low-FPR limitation of
the black-box variant. Prompt-hardening does **not** neutralize it (0.988 → 0.971).

![AUC per attack and target](assets/auc_barchart.png)

📊 **Full results, plots and methodology → [RESULTS.md](RESULTS.md)**

## What it measures

Three attacks implemented on top of the same black-box interface, plus baselines
described in the thesis:

- **RAG-MIA** (Anderson 2025) — prompt injection: asks the LLM whether the
  target document appears in the retrieved context. Black-box (parsed Yes/No) and
  **gray-box** (continuous `P(Yes)` from token logprobs).
- **S2MIA** (Li 2025) — signal from *BLEU* (answer↔document overlap) +
  *perplexity* (GPT-2 proxy, or the target's **native perplexity** in gray-box) +
  optional *cosine similarity*.
- **BudgetLeak** (Li 2025) — side-channel on the *generation budget* (max_tokens /
  `num_predict` depending on backend).

Attack metrics (`metrics` module): **AUC/ROC**, **TPR@1%FPR**, **PPV with
membership prior** (Bayesian form `PPV = π·TPR / (π·TPR + (1−π)·FPR)`),
**membership advantage** (`TPR − FPR`).

## Stack

Python 3.11, `uv` environment, everything runs **locally** by default on Apple
Silicon (MPS/Metal acceleration). Provider-agnostic since v0.2: swap LLM,
embedder, and perplexity scorer without touching attack code.

- **RAG:** LangChain + **Chroma** (in-memory or persistent vector DB) +
  pluggable embedding backend.
- **Target LLM:** provider-agnostic. Built-in:
  - `ollama` — `llama3.1:8b` local (primary, thesis default).
  - `azure_openai` — GPT-4o-mini via Azure OpenAI (API keys in `.env`).
  - `bedrock` — Claude via AWS Bedrock (AWS credentials in `.env`).
- **Embeddings:** `sentence_tf` (local MiniLM/BGE) or `openai_embed` (Azure).
- **Perplexity:** `gpt2` (local proxy, EN) or `hf_causal` (any HF causal LM,
  useful for Italian corpora via e.g. Minerva). Ollama v0.32.7 does not
  expose per-token logprobs via `/api/generate`, so a proxy LM is used.
- **PII:** regex (Italian tax codes, VAT, REA) + Italian NER (HuggingFace
  token-classification, default `rizzoaiacademy/rizzo-pii-0.3B`, configurable
  via env `MIARAG_NER_MODEL`).

## Structure

```
src/miarag/
  config.py               # Settings (frozen dataclass) + validate() + get_settings()
  pseudonymize.py         # PII gate: regex + NER → deterministic tokens, ≥9-digit backstop
  ingestion.py            # PDF/DOCX/MD/TXT → text → normalization → pseudonymization
  corpus.py               # chunking (char window, overlap) + member/non-member split
  rag.py                  # TargetRAG: black-box interface + DI (llm/embedder/ppl)
  providers/              # v0.2 pluggable backends
    base.py               # Protocols: LLMProvider, EmbeddingProvider, PerplexityScorer
    __init__.py           # factory + registry (lazy import)
    ollama.py             # OllamaProvider
    azure_openai.py       # AzureOpenAIProvider (retry + cost tracking)
    bedrock.py            # BedrockProvider (retry + cost tracking)
    embeddings/
      sentence_tf.py      # SentenceTransformer (local)
      openai_embed.py     # Azure OpenAI embeddings
    perplexity/
      gpt2.py             # GPT-2 (EN proxy)
      hf_causal.py        # Any HF causal LM (multilingual)
    _retry.py             # tenacity backoff for paid API
    _cost.py              # TRACKER singleton, per-provider call/char accounting
  attacks/
    s2mia.py              # S2MIA: BLEU + perplexity (+ optional cosine) → XGBoost score
    budgetleak.py         # BudgetLeak: Tri-Budget + Fuzzy C-Means zero-knowledge clustering
    rag_mia.py            # RAG-MIA (Anderson 2025): prompt injection black-box
  defenses.py             # text-based defenses: paraphrase, prompt hardening
  metrics.py              # AUC, TPR@FPR, PPV-with-prior, membership advantage
  plots.py                # multi-attack ROC
  dashboard_helpers.py    # pure logic for the dashboard (no streamlit import, testable)
tests/                    # offline suite (network-free), fake NER for tests
scripts/
  run_ingestion.py        # real corpus ingest (out of repo)
  run_attack.py           # attack orchestration (indexes ONLY members, saves scores CSV)
  run_eval.py             # evaluation (AUC/TPR@1%FPR/PPV, company vs person breakdown)
dashboard/
  app.py                  # Streamlit UI (5 sections: live attacks, metrics, PII, RAG, ingest)
```

## Setup

```bash
# dependencies
uv sync

# local target model (~5 GB pull, use stable WiFi)
ollama pull llama3.1:8b

# API keys (only if using paid targets) — never commit .env
cp .env.example .env   # then fill in
```

## Selecting providers (v0.2)

Provider choice via env or CLI. Default reproduces v0.1-thesis exactly
(Ollama + MiniLM + GPT-2).

```bash
# via .env
LLM_PROVIDER=ollama              # ollama | azure_openai | bedrock
EMBEDDING_PROVIDER=sentence_tf   # sentence_tf | openai_embed
PERPLEXITY_PROVIDER=gpt2         # gpt2 | hf_causal
PERPLEXITY_HF_MODEL=gpt2         # any HF causal LM if hf_causal

# via CLI (overrides env)
uv run python scripts/run_attack.py --llm azure_openai --embed openai_embed
```

`Settings.validate()` fails fast if the selected provider requires env vars
that are missing.

## Tests

Fully offline suite (no network, NER and Ollama mocked):

```bash
uv run pytest -q
```

96 passing, 1 skipped (a backcompat test isolated for a known macOS ARM
segfault when torch+xgboost+tqdm run together; runnable standalone). Attack
scripts set `OMP_NUM_THREADS=1` to avoid the dual-OpenMP segfault at scoring time.

## Real corpus ingestion

The corpus lives **outside the repo** (in `documenti/`, git-ignored). Ingest
pseudonymizes and writes JSONL to `data/processed/`:

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=src uv run python scripts/run_ingestion.py
```

## End-to-end pipeline

Full sequence (requires Ollama running: `ollama serve`):

```bash
# 1. corpus ingest (pseudonymize → data/processed/reports.jsonl)
HF_HUB_DISABLE_XET=1 PYTHONPATH=src uv run python scripts/run_ingestion.py

# 2. run the attacks (indexes ONLY members, saves results/scores_*.csv)
PYTHONPATH=src uv run python scripts/run_attack.py

# 3. evaluate (AUC/TPR@1%FPR/PPV, company vs person breakdown,
#    results/summary.csv + roc.png)
PYTHONPATH=src uv run python scripts/run_eval.py --prior 0.1

# defenses (security/utility trade-off): rerun attacks under defense
PYTHONPATH=src uv run python scripts/run_attack.py --defense paraphrase
PYTHONPATH=src uv run python scripts/run_attack.py --defense prompt_hardening
```

**On results.** Text-based defenses (paraphrase, prompt hardening) reduce the
textual signal for S2MIA (BLEU/perplexity) but **do not block BudgetLeak**,
which is a behavioral side-channel on the generation budget. The disaggregated
analysis by `has_person` (company vs person) highlights differential privacy
damage. Quantitative values (AUC, TPR, PPV) are produced by the pipeline
above and reported in the thesis.

## Dashboard (Streamlit)

Interactive presentation layer to explore the PoC. Requires
`uv run streamlit run dashboard/app.py`.

**Prerequisites:**
- `data/processed/reports.jsonl` ingested (see "Real corpus ingestion").
- Ollama running (`ollama serve`) for live attacks and RAG exploration.

**Sections:**
1. **Live Attacks** — S2MIA/BudgetLeak on Ollama in real time (slow;
   configurable limit on number of chunks).
2. **Metrics & Charts** — displays `results/summary.csv` and `roc.png`
   (from `run_eval.py`).
3. **PII Gate** — pseudonymization demo (user input → output with tokens;
   regex-only by default for speed, NER opt-in).
4. **RAG Explorer** — black-box query → answer + retrieved chunk IDs.
5. **Document Ingest** — upload PDF/DOCX/MD/TXT → pseudonymization + append
   to `reports.jsonl`.

The sidebar also exposes a **Provider (advanced)** expander to switch LLM /
embedder / perplexity backends at runtime.

## Status

| Component | Status |
|---|---|
| Config + scaffold | ✅ |
| PII pseudonymization (ethical gate) | ✅ — 0 residual PII on real ingest (5 PDFs) |
| PDF ingestion | ✅ |
| Chunking + member split | ✅ |
| TargetRAG (black-box) | ✅ |
| Attack metrics | ✅ |
| S2MIA attack | ✅ |
| BudgetLeak attack | ✅ |
| RAG-MIA attack (Anderson 2025) | ✅ |
| Plots / charts | ✅ |
| End-to-end orchestration | ✅ |
| Defenses + trade-off | ✅ |
| Streamlit dashboard | ✅ |
| **v0.2 pluggable providers** | ✅ — LLM/embedder/PPL swappable |
| Retry + cost tracking (paid API) | ✅ |
| Persistent Chroma (opt-in) | ✅ |

## Notes

- Operational docs: [`OLLAMA.md`](OLLAMA.md) (local model setup),
  [`DS4_EC2.md`](DS4_EC2.md) (optional remote infrastructure),
  [`TODO.md`](TODO.md) (roadmap).
- PoC code versioning is separate from the thesis Vault.

## License

[MIT](LICENSE) © 2026 Daniele Tavolaro. Research proof-of-concept for a master's
thesis; provided "as is". The attack code is for defensive research and privacy
risk assessment on systems you own or are authorized to test.
