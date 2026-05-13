"""
llm_provider.py
---------------
Centralized LLM provider with automatic fallback:

- Groq tiered fallbacks (separate rate limits):
  1) llama-3.1-8b-instant
  2) gemma2-9b-it
- Final fallback: Google Gemini (ChatGoogleGenerativeAI) using gemini-2.0-flash

The fallback is triggered automatically on rate-limit style failures (HTTP 429).
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_GROQ_MODELS = ("llama-3.1-8b-instant", "gemma2-9b-it")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Detect common 429/rate-limit errors across LangChain providers.
    """
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "ratelimit" in msg:
        return True
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429


def _get_groq_llm(model: str):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in Backend/.env")
    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=model,
        temperature=0,
        max_tokens=1024,
    )


def _get_gemini_llm():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set in Backend/.env")
    from langchain_google_genai import ChatGoogleGenerativeAI

    # ChatGoogleGenerativeAI reads GOOGLE_API_KEY from env, but we also pass it explicitly.
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )


def get_llm():
    """
    Return an LLM instance.

    - Prefers Groq.
    - If a quick test call hits a 429/rate-limit error, tries the next Groq model.
    - If Groq is rate-limited, falls back to Gemini.
    - If Groq key is missing, falls back to Gemini.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # If Groq isn't configured, go straight to Gemini.
    if not GROQ_API_KEY:
        return _get_gemini_llm()

    test_messages = [
        SystemMessage(content="Reply with exactly: OK"),
        HumanMessage(content="OK"),
    ]

    for model in _GROQ_MODELS:
        groq = _get_groq_llm(model)
        try:
            _ = groq.invoke(test_messages)
            return groq
        except Exception as e:
            if _is_rate_limit_error(e):
                continue
            raise

    return _get_gemini_llm()


def chat_with_fallback(messages: Sequence[Any]) -> str:
    """
    Invoke Groq first; on any rate-limit / 429 error, try the next Groq model,
    then finally retry with Gemini.
    Returns response content as a string.
    """
    last_err: Exception | None = None

    # Tiered Groq models (separate limits)
    if GROQ_API_KEY:
        for model in _GROQ_MODELS:
            try:
                groq = _get_groq_llm(model)
                resp = groq.invoke(list(messages))
                content = (getattr(resp, "content", None) or "").strip()
                if not content:
                    raise RuntimeError("Empty response content from Groq.")
                return content
            except Exception as e:
                last_err = e
                if _is_rate_limit_error(e):
                    continue
                raise

    # Final fallback: Gemini
    try:
        gemini = _get_gemini_llm()
        resp = gemini.invoke(list(messages))
        content = (getattr(resp, "content", None) or "").strip()
        if not content:
            raise RuntimeError("Empty response content from Gemini.")
        return content
    except Exception as e:
        if last_err is not None:
            raise RuntimeError(
                f"All providers failed. Last Groq error: {last_err}. Gemini error: {e}"
            )
        raise

