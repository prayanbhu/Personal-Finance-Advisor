"""
Finance AI Agent — orchestration layer.

This is the heart of the "agentic" architecture:

    Streamlit UI -> FinanceAgent -> Groq Llama (tool-calling) -> Tools -> FinanceEngine

The agent maintains a bound-tools LLM and runs a standard tool-calling
loop: send the conversation to the model, and if it responds with
tool_calls, execute each tool against the deterministic finance engine,
feed the results back as ToolMessages, and let the model continue
reasoning until it produces a final natural-language answer. The LLM
never sees raw customer rows — only the structured JSON returned by a
tool it explicitly chose to call.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from finance_engine import FinanceEngine
from llm import get_llm
from memory import ConversationMemory
from prompts import AI_INSIGHTS_INSTRUCTION, REPORT_SUMMARY_INSTRUCTION, SYSTEM_PROMPT
from tools import build_tool_registry

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 6


class FinanceAgent:
    """A tool-calling AI agent scoped to one customer's FinanceEngine."""

    def __init__(self, engine: FinanceEngine, memory: ConversationMemory):
        self.engine = engine
        self.memory = memory
        self.tools = build_tool_registry(engine)
        self._tools_by_name = {t.name: t for t in self.tools}
        self._llm = get_llm()
        self._llm_with_tools = self._llm.bind_tools(self.tools)

    # ------------------------------------------------------------------ #
    # Internal: run the tool-calling loop
    # ------------------------------------------------------------------ #
    def _run_loop(self, messages: list) -> tuple[str, list[dict]]:
        """Execute the bind-tools reasoning loop and return (final_text, tool_trace)."""
        tool_trace: list[dict] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            response: AIMessage = self._llm_with_tools.invoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                return response.content, tool_trace

            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})
                tool = self._tools_by_name.get(tool_name)

                if tool is None:
                    result = {"error": f"Unknown tool '{tool_name}'"}
                else:
                    try:
                        result = tool.invoke(tool_args)
                    except Exception as exc:  # noqa: BLE001 - surface tool errors to the LLM
                        logger.exception("Tool '%s' raised an error", tool_name)
                        result = {"error": str(exc)}

                tool_trace.append({"tool": tool_name, "args": tool_args, "result": result})
                messages.append(
                    ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
                )

        # Safety net: force a final answer if the loop exhausted iterations.
        final = self._llm.invoke(messages)
        return final.content, tool_trace

    def _build_message_stack(self, extra_instruction: str | None = None) -> list:
        system_content = SYSTEM_PROMPT
        context = self.memory.context_summary()
        if context:
            system_content += f"\n\n{context}"

        messages: list = [SystemMessage(content=system_content)]
        messages.extend(self.memory.get_history())

        if extra_instruction:
            messages.append(HumanMessage(content=extra_instruction))
        return messages

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(self, user_message: str) -> tuple[str, list[dict]]:
        """Handle one user turn, updating memory, and returning (reply, tool_trace)."""
        self.memory.add_user_message(user_message)
        messages = self._build_message_stack()
        final_text, tool_trace = self._run_loop(messages)
        self.memory.add_ai_message(final_text)
        return final_text, tool_trace

    def run_instruction(self, instruction: str, persist_to_memory: bool = False) -> tuple[str, list[dict]]:
        """
        Run a one-off structured instruction (e.g. for AI Insights or Report tabs)
        without necessarily polluting the visible chat transcript.
        """
        if persist_to_memory:
            return self.chat(instruction)

        messages = self._build_message_stack(extra_instruction=instruction)
        return self._run_loop(messages)

    def generate_insights(self, month: str | None = None) -> tuple[str, list[dict]]:
        month_clause = f" for {month}" if month else ""
        instruction = f"{AI_INSIGHTS_INSTRUCTION}\nTarget month{month_clause}."
        return self.run_instruction(instruction)

    def generate_report_summary(self, month: str | None = None) -> tuple[str, list[dict]]:
        month_clause = f" for {month}" if month else ""
        instruction = f"{REPORT_SUMMARY_INSTRUCTION}\nTarget month{month_clause}."
        return self.run_instruction(instruction)
