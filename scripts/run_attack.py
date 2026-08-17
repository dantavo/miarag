# scripts/run_attack.py
import argparse
import csv, json
from pathlib import Path
from miarag.config import get_settings
from miarag.ingestion import ReportDoc
from miarag.corpus import chunk_documents, split_members, split_members_by_doc, Chunk
from miarag.rag import TargetRAG
from miarag.providers import build_llm, build_embedder, build_perplexity
from miarag.attacks.s2mia import s2mia_scores, s2mia_scores_native_ppl
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

def _save(path: Path, chunks, scores, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["chunk_id", "score", "label", "has_person"])
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

    if args.graybox:
        # Gray-box: usa logprob del target. Solo S2MIA e RAG-MIA hanno variante
        # gray-box; BudgetLeak è comportamentale (nessun logprob) e identico al
        # black-box → SALTATO qui per non duplicarlo. Suffisso _graybox (+ difesa).
        gb_suffix = "_graybox" + suffix
        attack_plan = [
            ("s2mia", s2mia_scores_native_ppl, gb_suffix),
            ("rag_mia", rag_mia_graybox_scores, gb_suffix),
        ]
    else:
        attack_plan = [
            ("s2mia", s2mia_scores, suffix),
            ("budgetleak", budgetleak_scores, suffix),
            ("rag_mia", rag_mia_scores, suffix),
        ]

    from pathlib import Path as _Path
    out_dir = _Path(args.results_dir) if args.results_dir else s.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"       output dir: {out_dir}", flush=True)

    for name, fn, suf in attack_plan:
        t0 = _time.time()
        print(f"       ▶ {name}{suf} starting on {len(targets)} chunks...", flush=True)
        scores, labels = fn(rag, targets)
        _save(out_dir / f"scores_{name}{suf}.csv", targets, scores, labels)
        dt = _time.time() - t0
        print(f"       ✓ {name}{suf}: saved {len(scores)} scores ({dt:.1f}s, {dt/len(targets):.2f}s/chunk)", flush=True)

if __name__ == "__main__":
    main()
