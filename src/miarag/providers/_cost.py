# src/miarag/providers/_cost.py
"""Cost / token accounting per provider paid API.

Design: singleton globale `TRACKER` accumula (calls, prompt_chars, completion_chars).
Ogni provider paid chiama TRACKER.record(...) dopo generate().

Nota: usa chars come proxy (evita dipendenza tiktoken). Per stime precise su Azure:
approx tokens ≈ chars / 4 per EN, ≈ chars / 3.5 per IT.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class UsageStats:
    calls: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    by_provider: dict[str, dict[str, int]] = field(default_factory=dict)

    def approx_tokens(self, chars_per_token: float = 3.8) -> int:
        return int((self.prompt_chars + self.completion_chars) / chars_per_token)


class _Tracker:
    def __init__(self):
        self._stats = UsageStats()
        self._lock = Lock()

    def record(self, provider: str, prompt: str, completion: str) -> None:
        with self._lock:
            self._stats.calls += 1
            self._stats.prompt_chars += len(prompt)
            self._stats.completion_chars += len(completion)
            p = self._stats.by_provider.setdefault(
                provider, {"calls": 0, "prompt_chars": 0, "completion_chars": 0}
            )
            p["calls"] += 1
            p["prompt_chars"] += len(prompt)
            p["completion_chars"] += len(completion)

    def snapshot(self) -> UsageStats:
        with self._lock:
            # shallow copy sufficiente per read-only.
            return UsageStats(
                calls=self._stats.calls,
                prompt_chars=self._stats.prompt_chars,
                completion_chars=self._stats.completion_chars,
                by_provider={k: dict(v) for k, v in self._stats.by_provider.items()},
            )

    def reset(self) -> None:
        with self._lock:
            self._stats = UsageStats()


TRACKER = _Tracker()
