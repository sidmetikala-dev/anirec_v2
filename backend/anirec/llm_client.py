"""Provider configuration and OpenAI-compatible completion adapters."""
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from groq import Groq, RateLimitError

logger = logging.getLogger(__name__)

def _retry_delay(exc):
    """Read provider retry timing without retrying long daily-limit waits."""
    headers = exc.response.headers
    value = headers.get("retry-after")
    try:
        delay = float(value) if value is not None else None
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            delay = None
    if delay is None:
        match = re.search(r"try again in\s+(\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
        delay = float(match.group(1)) if match else None
    return max(0, delay) if delay is not None and math.isfinite(delay) else None


def create_groq_completion(client, **kwargs):
    if kwargs.get("model") in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
        effort = os.getenv("GROQ_REASONING_EFFORT", "medium")
        if effort not in {"low", "medium", "high"}:
            raise ValueError("GROQ_REASONING_EFFORT must be low, medium, or high")
        kwargs.setdefault("reasoning_effort", effort)
        kwargs["max_completion_tokens"] = max(kwargs.get("max_completion_tokens", 0), 2048)
    # At most two retries and 30 seconds total waiting per API call.
    waited = 0.0
    for attempt in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            delay = _retry_delay(exc)
            if attempt == 2 or delay is None or waited + delay + 0.1 > 30:
                raise
            delay += 0.1
            logger.info("Groq rate limited; retrying in %.2f seconds", delay)
            time.sleep(delay)
            waited += delay


class OllamaClient:
    """Translate shared chat arguments to Ollama's supported API."""

    provider = "ollama"

    def __init__(self, client):
        self.client = client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        kwargs.pop("parallel_tool_calls", None)
        kwargs.pop("tool_choice", None)
        kwargs.pop("reasoning_effort", None)
        if "max_completion_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        return self.client.chat.completions.create(**kwargs)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class GeminiClient(OllamaClient):
    """Use Gemini's OpenAI-compatible endpoint with the shared chat interface."""

    provider = "gemini"

    def create(self, **kwargs):
        kwargs.pop("parallel_tool_calls", None)
        if "max_completion_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        return self.client.chat.completions.create(**kwargs)


def provider_config():
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in {"groq", "ollama", "gemini"}:
        raise ValueError("LLM_PROVIDER must be groq, ollama, or gemini")
    default = {"groq": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
               "ollama": "qwen2.5:7b",
               "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")}[provider]
    return provider, os.getenv("LLM_MODEL") or default


def create_client(provider):
    if provider == "groq":
        return Groq(api_key=os.environ["GROQ_API_KEY"], timeout=30.0, max_retries=0)
    if provider == "gemini":
        from openai import OpenAI
        return GeminiClient(OpenAI(api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=60.0, max_retries=0))
    if provider == "ollama":
        from openai import OpenAI
        return OllamaClient(OpenAI(base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
                                   api_key="ollama", timeout=120.0, max_retries=0))
    raise ValueError("LLM_PROVIDER must be groq, ollama, or gemini")


def create_completion(client, **kwargs):
    # Groq clients (including recorded clients) retain per-attempt evaluation records.
    if isinstance(client, OllamaClient) or getattr(client, "provider", None) in {"ollama", "gemini"}:
        return client.chat.completions.create(**kwargs)
    return create_groq_completion(client, **kwargs)
