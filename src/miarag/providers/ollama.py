# src/miarag/providers/ollama.py
"""OllamaProvider: wrap Ollama HTTP API con timeout + retry espliciti.

Motivazione: langchain_ollama.OllamaLLM non setta timeout HTTP. Su chiamate lunghe
con keep-alive, connessioni possono finire in CLOSE_WAIT lato client (server ha
chiuso, client non ha noticed) → deadlock indefinito. Osservato in produzione
durante attacchi su corpus grandi (500 chunk × BudgetLeak).

Fix: uso httpx diretto con read timeout e connect timeout espliciti, + retry
esponenziale su errori transient.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OllamaProvider:
    base_url: str
    model: str
    name: str = "ollama"
    # Timeout in secondi. read=180s copre worst-case: llama3.1:8b Q4 su M5 con
    # num_predict=256 in contesto RAG ≈ 30-60s. 180s dà ampio margine ma rileva
    # deadlock reali entro 3 minuti (non 8 ore come precedente).
    connect_timeout: float = 10.0
    read_timeout: float = 180.0
    max_retries: int = 3

    def __post_init__(self):
        import httpx
        # Client persistente con timeout espliciti. Meglio di ricreare connessione
        # ad ogni chiamata (overhead TCP handshake) e meglio di connessioni
        # long-lived senza timeout (deadlock su CLOSE_WAIT).
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=self.connect_timeout,
                read=self.read_timeout,
                write=self.read_timeout,
                pool=self.read_timeout,
            ),
        )

    def generate(self, prompt: str, max_tokens: int) -> str:
        import time
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post(
                    "/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": max_tokens},
                    },
                )
                r.raise_for_status()
                return r.json().get("response", "")
            except Exception as e:
                last_err = e
                # Backoff esponenziale: 2s, 4s, 8s. Chiude client e ricrea:
                # elimina eventuali connessioni HTTP in stato inconsistente.
                if attempt < self.max_retries - 1:
                    try:
                        self._client.close()
                    except Exception:
                        pass
                    import httpx
                    self._client = httpx.Client(
                        base_url=self.base_url,
                        timeout=httpx.Timeout(
                            connect=self.connect_timeout,
                            read=self.read_timeout,
                            write=self.read_timeout,
                            pool=self.read_timeout,
                        ),
                    )
                    time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"Ollama generate failed after {self.max_retries} attempts: {last_err}")

    def supports_logprobs(self) -> bool:
        # Ollama /api/generate non espone per-token logprobs a v0.32.7.
        return False

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass
