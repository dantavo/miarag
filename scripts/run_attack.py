# scripts/run_attack.py
import os
# macOS ARM: torch (gpt2 perplexity) e xgboost (S²MIA-M) caricano DUE runtime
# OpenMP nello stesso processo → segfault nativo dentro cross_val_predict (bypassa
# try/except: attacco muore SENZA scrivere il CSV). OMP_NUM_THREADS=1 lo previene.
# Deve essere impostato PRIMA di importare torch/xgboost. Impatto perf ~nullo:
# gpt2 gira su MPS, XGBoost su 1407×2 feature è banale.
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import csv, json
from pathlib import Path
from miarag.config import get_settings
from miarag.ingestion import ReportDoc
from miarag.corpus import chunk_documents, split_members, split_members_by_doc, Chunk
from miarag.rag import TargetRAG
from miarag.providers import build_llm, build_embedder, build_perplexity
from miarag.attacks.s2mia import s2mia_scores_with_feats
from miarag.attacks.budgetleak import budgetleak_scores
from miarag.attacks.rag_mia import rag_mia_scores, rag_mia_graybox_scores
from miarag.defenses import apply_defense

def _load_reports(path: Path) -> list[ReportDoc]:
    docs = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        docs.append(ReportDoc(**d))
    return docs

def build_pipeline(reports_jsonl: Path, rag, split: str = "chunk") -> tuple[list[Chunk], list[Chunk]]:
    docs = _load_reports(reports_jsonl)
    chunks = chunk_documents(docs)
    # split="doc": document-level membership (literature-standard, no sibling-chunk
    # leakage). split="chunk": legacy chunk-level shuffle (v0.1-thesis, Aug-12 numbers).
    if split == "doc":
        members, non_members = split_members_by_doc(chunks, 0.5, 42)
    else:
        members, non_members = split_members(chunks, 0.5, 42)
    rag.index(members)          # SOLO i membri sono indicizzati
    return members, non_members

def _save(path: Path, chunks, scores, labels, feats=None):
    """Salva gli score. Se feats (lista di {bleu, ppl}) è fornita, aggiunge le
    colonne bleu,ppl per verificabilità (usato da S2MIA). Altrimenti scrive le
    4 colonne storiche (retrocompat con run_eval._load, che legge per nome)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        if feats is not None:
            w.writerow(["chunk_id", "score", "label", "has_person", "bleu", "ppl"])
            for c, s, l, ft in zip(chunks, scores, labels, feats):
                w.writerow([c.chunk_id, s, l, int(c.has_person), ft["bleu"], ft["ppl"]])
        else:
            w.writerow(["chunk_id", "score", "label", "has_person"])
            for c, s, l in zip(chunks, scores, labels):
                w.writerow([c.chunk_id, s, l, int(c.has_person)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense", choices=["none", "paraphrase", "prompt_hardening"], default="none")
    parser.add_argument("--llm", choices=["ollama", "azure_openai", "bedrock"], default=None,
                        help="Override LLM_PROVIDER env. Default: da Settings.")
    parser.add_argument("--embed", choices=["sentence_tf", "openai_embed"], default=None,
                        help="Override EMBEDDING_PROVIDER env.")
    parser.add_argument("--ppl", choices=["gpt2", "hf_causal"], default=None,
                        help="Override PERPLEXITY_PROVIDER env.")
    parser.add_argument("--split", choices=["chunk", "doc"], default="chunk",
                        help="Membership split: 'chunk' (legacy, v0.1-thesis) or 'doc' "
                             "(document-level, no sibling-chunk leakage — recommended).")
    parser.add_argument("--graybox", action="store_true",
                        help="Gray-box S2MIA (native perplexity from target logprobs) + "
                             "RAG-MIA (continuous P(Yes) from first-token logprobs). "
                             "Requires an LLM provider exposing logprobs (e.g. azure_openai). "
                             "Writes scores_{s2mia,rag_mia}_graybox.csv; budgetleak unchanged.")
    parser.add_argument("--results-dir", default=None,
                        help="Directory di output per gli scores (default: results/). "
                             "Usa una dir separata per eseguire run in parallelo senza "
                             "sovrascrivere i CSV (es. --results-dir results_ollama44).")
    parser.add_argument("--attacks", default=None,
                        help="Sottoinsieme di attacchi da eseguire (csv: s2mia,budgetleak,rag_mia). "
                             "Default: tutti. Utile per ri-eseguire solo quelli falliti.")
    args = parser.parse_args()

    # CLI override → env → default. get_settings() rilegge env.
    import os
    if args.llm: os.environ["LLM_PROVIDER"] = args.llm
    if args.embed: os.environ["EMBEDDING_PROVIDER"] = args.embed
    if args.ppl: os.environ["PERPLEXITY_PROVIDER"] = args.ppl

    s = get_settings()
    s.validate()   # fail-fast su chiavi mancanti

    print(f"[1/6] building llm={s.llm_provider}...", flush=True)
    llm = build_llm(s)
    print(f"[2/6] building embedder={s.embedding_provider}...", flush=True)
    embedder = build_embedder(s)
    print(f"[3/6] building ppl={s.perplexity_provider}...", flush=True)
    ppl = build_perplexity(s)
    print(f"[4/6] constructing TargetRAG (top_k={s.top_k})...", flush=True)
    rag = TargetRAG(llm=llm, embedder=embedder, ppl=ppl, top_k=s.top_k)

    print(f"provider: llm={llm.name} embed={embedder.name} ppl={ppl.name}", flush=True)

    # Apply defense wrapper (NOTE: text-based defenses knock down S2MIA BLEU/perplexity signal
    # but do NOT stop BudgetLeak, a behavioral side-channel — key finding of the thesis)
    rag = apply_defense(rag, args.defense)

    print(f"[5/6] loading reports + chunking + indexing members (split={args.split})...", flush=True)
    members, non_members = build_pipeline(s.data_dir / "processed" / "reports.jsonl", rag, split=args.split)
    targets = members + non_members
    print(f"       corpus: {len(members)} members + {len(non_members)} non-members = {len(targets)} total", flush=True)

    # Filename logic: plain name for --defense none (keeps run_eval.py contract),
    # suffix for non-none defenses
    suffix = "" if args.defense == "none" else f"_{args.defense}"

    print(f"[6/6] running attacks (defense={args.defense}, graybox={args.graybox})...", flush=True)
    import time as _time

    # is_s2mia flag: S2MIA usa s2mia_scores_with_feats → (scores, labels, feats, kept),
    # salva anche bleu/ppl. Gli altri usano scored_loop → (scores, labels).
    if args.graybox:
        # Gray-box: usa logprob del target. Solo S2MIA e RAG-MIA hanno variante
        # gray-box; BudgetLeak è comportamentale (nessun logprob) e identico al
        # black-box → SALTATO qui per non duplicarlo. Suffisso _graybox (+ difesa).
        gb_suffix = "_graybox" + suffix
        attack_plan = [
            ("s2mia", lambda rag, ch: s2mia_scores_with_feats(rag, ch, graybox=True), gb_suffix, True),
            ("rag_mia", rag_mia_graybox_scores, gb_suffix, False),
        ]
    else:
        attack_plan = [
            ("s2mia", s2mia_scores_with_feats, suffix, True),
            ("budgetleak", budgetleak_scores, suffix, False),
            ("rag_mia", rag_mia_scores, suffix, False),
        ]

    from pathlib import Path as _Path
    out_dir = _Path(args.results_dir) if args.results_dir else s.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"       output dir: {out_dir}", flush=True)

    # Filtro opzionale sugli attacchi (--attacks s2mia,rag_mia).
    if args.attacks:
        wanted = {a.strip() for a in args.attacks.split(",") if a.strip()}
        attack_plan = [t for t in attack_plan if t[0] in wanted]
        print(f"       attacchi selezionati: {[t[0] for t in attack_plan]}", flush=True)

    for name, fn, suf, is_s2mia in attack_plan:
        t0 = _time.time()
        print(f"       ▶ {name}{suf} starting on {len(targets)} chunks...", flush=True)
        try:
            if is_s2mia:
                # S²MIA-M: ritorna anche feature grezze e i chunk effettivamente valutati.
                scores, labels, feats, kept = fn(rag, targets)
                _save(out_dir / f"scores_{name}{suf}.csv", kept, scores, labels, feats=feats)
            else:
                scores, labels = fn(rag, targets)
                _save(out_dir / f"scores_{name}{suf}.csv", targets, scores, labels)
            dt = _time.time() - t0
            print(f"       ✓ {name}{suf}: saved {len(scores)} scores ({dt:.1f}s, {dt/len(targets):.2f}s/chunk)", flush=True)
        except Exception as e:  # resiliente: un attacco fallito non blocca gli altri
            dt = _time.time() - t0
            print(f"       ✗ {name}{suf}: FALLITO dopo {dt:.0f}s — {type(e).__name__}: {str(e)[:200]}", flush=True)

if __name__ == "__main__":
    main()
