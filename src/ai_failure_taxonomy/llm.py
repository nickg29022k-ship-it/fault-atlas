"""LLM backend abstraction.

`get_backend(name)` returns an object with
`complete(prompt: str, system: str | None = None, temperature: float = 0.0) -> str`.

Backends:
  mock       deterministic, offline (default)
  ollama     local Ollama server at OLLAMA_HOST (default http://localhost:11434), model OLLAMA_MODEL
  anthropic  Messages API, needs ANTHROPIC_API_KEY, model ANTHROPIC_MODEL
  openai     Chat Completions API, needs OPENAI_API_KEY, model OPENAI_MODEL

Network backends use plain httpx and raise BackendUnavailable with a readable
message when the service or credentials are missing.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import httpx

DEFAULT_TIMEOUT = float(os.environ.get("AFT_HTTP_TIMEOUT", "120"))


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot be reached or is not configured."""


class LLMBackend(Protocol):
    name: str

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str: ...


class MockBackend:
    """Deterministic offline backend.

    For failure-classification prompts (system prompt contains the task
    marker) it simulates an LLM judge: it reads the item out of the prompt,
    applies shallow cue-based judgments, then injects seeded noise (missed
    labels, spurious labels, varied output formatting) so downstream parsing
    and ensembling face realistic conditions. For any other prompt it returns
    a deterministic placeholder string.
    """

    name = "mock"

    def __init__(self, seed: int = 13):
        self.seed = seed

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        from .mock_judge import TASK_MARKER, judge

        if (system and TASK_MARKER in system) or TASK_MARKER in prompt:
            return judge(prompt, seed=self.seed)
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return f"[mock completion {digest}] {prompt[:60]}"


def _post(url: str, payload: dict, headers: dict, service: str) -> dict:
    try:
        r = httpx.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    except httpx.ConnectError as e:
        raise BackendUnavailable(f"{service}: cannot connect to {url} ({e}). Is the service running?") from e
    except httpx.TimeoutException as e:
        raise BackendUnavailable(f"{service}: request to {url} timed out") from e
    except httpx.HTTPError as e:
        raise BackendUnavailable(f"{service}: HTTP error talking to {url}: {e}") from e
    if r.status_code in (401, 403):
        raise BackendUnavailable(f"{service}: authentication failed (HTTP {r.status_code}); check your API key")
    if r.status_code == 404:
        raise BackendUnavailable(f"{service}: endpoint or model not found (HTTP 404): {r.text[:200]}")
    if r.status_code >= 400:
        raise BackendUnavailable(f"{service}: HTTP {r.status_code}: {r.text[:300]}")
    try:
        return r.json()
    except ValueError as e:
        raise BackendUnavailable(f"{service}: response was not JSON: {r.text[:200]}") from e


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
        if system:
            payload["system"] = system
        data = _post(f"{self.host}/api/generate", payload, {}, "ollama")
        if "response" not in data:
            raise BackendUnavailable(f"ollama: unexpected response shape: {str(data)[:200]}")
        return data["response"]


class AnthropicBackend:
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 1024):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise BackendUnavailable("anthropic: ANTHROPIC_API_KEY is not set")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.max_tokens = max_tokens

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = _post(self.url, payload, headers, "anthropic")
        texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        if not texts:
            raise BackendUnavailable(f"anthropic: no text in response (stop_reason={data.get('stop_reason')})")
        return "".join(texts)


class OpenAIBackend:
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise BackendUnavailable("openai: OPENAI_API_KEY is not set")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        data = _post(self.url, payload, {"Authorization": f"Bearer {self.api_key}"}, "openai")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise BackendUnavailable(f"openai: unexpected response shape: {str(data)[:200]}") from e


BACKENDS = {
    "mock": MockBackend,
    "ollama": OllamaBackend,
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
}


def get_backend(name: str = "mock", **kwargs) -> LLMBackend:
    try:
        cls = BACKENDS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}") from None
    return cls(**kwargs)
