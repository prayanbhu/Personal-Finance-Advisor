"""
Conversation and context memory for the Finance AI Agent.

Two layers are maintained:

1. **Buffer memory** — the raw turn-by-turn chat history, replayed to
   the LLM on every call so it has full conversational context.
2. **Structured context memory** — key facts extracted from the
   conversation (selected customer, stated goals, savings targets,
   hypothetical figures the user mentions) that should persist and
   inform reasoning even many turns later, without relying on the LLM
   to re-scan the entire chat transcript.

This is intentionally a small, dependency-free implementation (rather
than `langchain.memory.ConversationBufferMemory`, which is deprecated
upstream) so behavior stays predictable and easy to extend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# Matches phrases like "my income is 90000", "income is ₹90,000", "I earn 90k"
_INCOME_PATTERN = re.compile(
    r"(?:my\s+income\s+is|i\s+earn|income\s+of)\s*[₹rs\.]*\s*([\d,]+(?:\.\d+)?)\s*(k)?",
    re.IGNORECASE,
)
# Matches phrases like "save 25000 every month", "savings target of 20000"
_SAVINGS_GOAL_PATTERN = re.compile(
    r"(?:save|saving\s+target\s+of|goal\s+of\s+saving)\s*[₹rs\.]*\s*([\d,]+(?:\.\d+)?)\s*(k)?",
    re.IGNORECASE,
)


def _parse_amount(raw: str, is_k: str | None) -> float:
    value = float(raw.replace(",", ""))
    if is_k:
        value *= 1000
    return value


@dataclass
class ConversationMemory:
    """Per-session memory scoped to one customer's chat with the agent."""

    customer_id: str | None = None
    messages: list[BaseMessage] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Buffer memory
    # ------------------------------------------------------------------ #
    def add_user_message(self, text: str) -> None:
        self._extract_context(text)
        self.messages.append(HumanMessage(content=text))

    def add_ai_message(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))

    def get_history(self) -> list[BaseMessage]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()
        self.context.clear()

    # ------------------------------------------------------------------ #
    # Structured context extraction
    # ------------------------------------------------------------------ #
    def _extract_context(self, text: str) -> None:
        income_match = _INCOME_PATTERN.search(text)
        if income_match:
            self.context["stated_income"] = _parse_amount(*income_match.groups())

        goal_match = _SAVINGS_GOAL_PATTERN.search(text)
        if goal_match:
            self.context["savings_goal"] = _parse_amount(*goal_match.groups())

    def set_goal(self, key: str, value) -> None:
        self.context[key] = value

    def context_summary(self) -> str:
        """Human-readable context block injected into the LLM prompt."""
        if not self.context:
            return ""
        lines = ["Known context from earlier in this conversation:"]
        if "stated_income" in self.context:
            lines.append(f"- Customer mentioned a hypothetical/stated income of ₹{self.context['stated_income']:,.0f}")
        if "savings_goal" in self.context:
            lines.append(f"- Customer mentioned a savings goal of ₹{self.context['savings_goal']:,.0f} per month")
        for k, v in self.context.items():
            if k not in ("stated_income", "savings_goal"):
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)


class MemoryStore:
    """Holds one ConversationMemory per customer_id so switching customers
    in the UI doesn't leak context between unrelated sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationMemory] = {}

    def get(self, customer_id: str) -> ConversationMemory:
        if customer_id not in self._sessions:
            self._sessions[customer_id] = ConversationMemory(customer_id=customer_id)
        return self._sessions[customer_id]

    def reset(self, customer_id: str) -> None:
        self._sessions[customer_id] = ConversationMemory(customer_id=customer_id)
