# src/miarag/providers/_mps.py
"""Lock globale per le operazioni su MPS/torch (embedder MiniLM, gpt2 perplexity).

PyTorch MPS non è thread-safe per accessi concorrenti dallo stesso processo.
Quando l'estrazione feature S2MIA gira multi-thread (MIARAG_S2MIA_WORKERS>1)
per sfruttare il continuous-batching di Ollama, le chiamate MPS devono
serializzare. Sono <5% del tempo per-chunk (la generazione HTTP verso Ollama,
~95%, resta concorrente), quindi il lock NON è un collo di bottiglia.

Se i worker == 1 (default) il lock è di fatto sempre libero: nessun overhead.
"""
import threading

MPS_LOCK = threading.RLock()
