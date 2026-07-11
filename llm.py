"""
Groq LLM client wrapper.

Isolates all direct dependency on `langchain_groq` so the rest of the
app talks to a single `get_llm()` factory instead of instantiating
ChatGroq clients ad hoc.
"""

from __future__ import annotations

import logging

from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


class MissingAPIKeyError(RuntimeError):
    """Raised when GROQ_API_KEY is not configured."""


def get_llm(temperature: float | None = None) -> ChatGroq:
    """
    Return a configured ChatGroq client.

    Raises MissingAPIKeyError early (rather than letting the Groq SDK
    raise an opaque auth error later) so the UI can show a clear setup
    message.
    """
    if not GROQ_API_KEY:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Create a .env file with "
            "GROQ_API_KEY=your_key_here (see .env.example)."
        )

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=LLM_MAX_TOKENS,
    )
