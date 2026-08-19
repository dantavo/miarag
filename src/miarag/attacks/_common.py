# src/miarag/attacks/_common.py
"""Helper condiviso: loop attacco con progress-logging + ETA reale + resilienza per-chunk.

Ogni attacco passa una `score_fn(chunk) -> float`. Il loop:
- logga avanzamento ogni ~5% (i/N, %, elapsed, ETA stimata dal rate reale, skip)
- salta il chunk se `score_fn` solleva (blip di rete) senza uccidere l'attacco
- mantiene scores/labels allineati
"""
from __future__ import annotations
import time
from typing import Callable
from miarag.corpus import Chunk


def scored_loop(name: str, chunks: list[Chunk], score_fn: Callable[[Chunk], float],
                log_every: int | None = None) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    skipped = 0
    n = len(chunks)
    if n == 0:
        return scores, labels
    step = log_every or max(1, n // 20)   # ~5% steps
    t0 = time.time()
    print(f"[{name}] start su {n} chunk", flush=True)
    for i, c in enumerate(chunks, 1):
        try:
            scores.append(float(score_fn(c)))
            labels.append(int(c.is_member))
        except Exception as e:  # resilienza: blip su un chunk non uccide l'attacco
            skipped += 1
            print(f"[{name}] skip {c.chunk_id}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        if i % step == 0 or i == n:
            el = time.time() - t0
            rate = el / i
            eta = rate * (n - i)
            print(f"[{name}] {i}/{n} ({100*i//n}%) · {el:.0f}s trascorsi · "
                  f"ETA ~{eta/60:.1f}min · {rate:.2f}s/chunk · {skipped} skip", flush=True)
    if skipped:
        print(f"[{name}] TOTale {skipped}/{n} chunk saltati per errori transitori", flush=True)
    return scores, labels
