# src/miarag/providers/_retry.py
"""Retry helpers per provider paid API. Backoff esponenziale su errori transient.

Uso:
    from miarag.providers._retry import api_retry

    @api_retry
    def call(): ...
"""
from __future__ import annotations
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)
import logging

log = logging.getLogger(__name__)

# Errori generici da riprovare. Provider-specifici possono estendere questa tupla.
_RETRYABLE = (ConnectionError, TimeoutError)


def api_retry(fn):
    """Decorator: 3 tentativi, backoff esponenziale 2s→8s."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )(fn)
