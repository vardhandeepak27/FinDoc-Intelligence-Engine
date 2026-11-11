"""One LLM client for every provider.

Groq, Gemini and Ollama all offer an OpenAI-compatible API, so we use the
`openai` Python package for all three and change only the base URL, API
key and model name. Switching provider is a config change, not a code
change, and the rest of the app never knows which provider it's using.

Free tiers have rate limits (Groq's free tier is ~8,000 tokens/minute for
gpt-oss-120b). The SDK already retries "429 Too Many Requests" with
exponential backoff and honours the server's Retry-After header; we just
raise the retry count.

Smoke test:  python src\\llm.py
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": None,  # set FINDOC_LLM_MODEL (model names change often)
    },
    "ollama": {
        "base_url": None,       # from settings.ollama_base_url
        "default_model": None,  # whichever model you've pulled locally
    },
}

REASONING_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float
    usage: dict = field(default_factory=dict)


def _api_key() -> str:
    provider = settings.llm_provider
    if provider == "ollama":
        return "ollama"  # Ollama ignores the key, but the SDK requires one
    secret = settings.groq_api_key if provider == "groq" else settings.gemini_api_key
    if secret is None:
        raise RuntimeError(
            f"No API key for '{provider}'. Set FINDOC_{provider.upper()}_API_KEY in .env"
        )
    return secret.get_secret_value()  # the ONLY place the key is unwrapped


def model_name() -> str:
    name = settings.llm_model or PROVIDERS[settings.llm_provider]["default_model"]
    if not name:
        raise RuntimeError(
            f"Set FINDOC_LLM_MODEL for provider '{settings.llm_provider}'."
        )
    return name


@lru_cache
def get_client() -> OpenAI:
    provider = PROVIDERS[settings.llm_provider]
    return OpenAI(
        base_url=provider["base_url"] or settings.ollama_base_url,
        api_key=_api_key(),
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


class TokenRateLimiter:
    """Client-side rate limiting: stay under the provider's tokens/minute.

    The SDK already retries after a 429 error, but hitting the limit wastes
    a request and waits blindly. Here we remember how many tokens we used
    in the last 60 seconds and pause BEFORE a request that would exceed
    the budget. (A sliding-window limiter; token buckets are a close cousin.)
    """

    def __init__(self, tokens_per_minute: int | None):
        self.limit = tokens_per_minute
        self.window: deque[tuple[float, int]] = deque()

    def wait(self, estimated_tokens: int) -> None:
        if not self.limit:
            return
        while True:
            now = time.monotonic()
            while self.window and now - self.window[0][0] > 60:
                self.window.popleft()
            used = sum(tokens for _, tokens in self.window)
            if used + estimated_tokens <= self.limit or not self.window:
                return
            pause = 60 - (now - self.window[0][0]) + 0.5
            logger.info("Rate limit: %d tokens used in the last minute, waiting %.0fs",
                        used, pause)
            time.sleep(pause)

    def record(self, tokens: int) -> None:
        self.window.append((time.monotonic(), tokens))


_limiters: dict[str, TokenRateLimiter] = {}


def _limiter(model: str) -> TokenRateLimiter:
    # Groq's limits are per model, so each model gets its own window
    if model not in _limiters:
        _limiters[model] = TokenRateLimiter(
            settings.model_tokens_per_minute.get(model, settings.llm_tokens_per_minute)
        )
    return _limiters[model]


def chat(
    messages: list[dict],
    max_tokens: int | None = None,
    model: str | None = None,
    json_mode: bool = False,
) -> LLMResponse:
    """Send a chat request and return the answer text plus metadata.

    model:     override the configured model (e.g. a different judge model)
    json_mode: ask the provider to return a single JSON object
    """
    model = model or model_name()
    params = {
        "model": model,
        "messages": messages,
        "temperature": settings.llm_temperature,
        "max_tokens": max_tokens or settings.llm_max_tokens,
    }
    if model.startswith("qwen/"):
        # Qwen reasoning models: hide the thinking from the reply. Groq
        # REQUIRES "hidden" or "parsed" with JSON mode (400 error otherwise).
        params["reasoning_effort"] = "low"
        params["extra_body"] = {"reasoning_format": "hidden"}
    elif model in REASONING_MODELS:
        # gpt-oss "thinks" before answering. Low effort is enough for
        # reading numbers out of sources, and saves tokens (= rate limit).
        params["reasoning_effort"] = settings.llm_reasoning_effort
        # include_reasoning is Groq-specific, not part of the OpenAI API, so
        # the SDK rejects it as a normal argument. extra_body passes
        # provider-specific fields straight into the request.
        params["extra_body"] = {"include_reasoning": False}
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    limiter = _limiter(model)
    prompt_chars = sum(len(m["content"]) for m in messages)
    limiter.wait(int(prompt_chars / 3.5) + 300)

    started = time.perf_counter()
    response = get_client().chat.completions.create(**params)
    latency = time.perf_counter() - started

    usage = response.usage.model_dump() if response.usage else {}
    limiter.record(usage.get("total_tokens") or int(prompt_chars / 3.5))
    logger.info(
        "LLM %s: %.1fs, %s prompt + %s completion tokens",
        model, latency, usage.get("prompt_tokens"), usage.get("completion_tokens"),
    )
    text = response.choices[0].message.content or ""
    if response.choices[0].finish_reason == "length":
        logger.warning("Answer was cut off at max_tokens=%s", params["max_tokens"])
    return LLMResponse(text=text.strip(), model=model, latency_s=latency, usage=usage)


if __name__ == "__main__":
    import sys

    from logging_setup import setup_logging

    setup_logging()
    if "--models" in sys.argv:
        # The docs list what a provider offers; this lists what YOUR key can use
        for m in sorted(get_client().models.list().data, key=lambda m: m.id):
            print(m.id)
        sys.exit()

    reply = chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=200)
    print(f"Provider: {settings.llm_provider} | model: {reply.model}")
    print(f"Reply: {reply.text!r} | {reply.latency_s:.2f}s | usage: {reply.usage}")
