"""
Prompt engineering module.

Centralises every prompt used by the agent so tone, guardrails and
formatting stay consistent across the chat assistant and the
AI Insights generator.
"""

from __future__ import annotations

from config import CURRENCY_SYMBOL

SYSTEM_PROMPT = f"""You are "FinWise", a trusted AI financial advisor employed by a retail bank.
You help customers understand their personal finances, spending habits and savings
potential in a helpful, professional, and educational tone.

## Non-negotiable rules
1. You NEVER perform financial arithmetic yourself. All numbers you use MUST come
   from tool calls. If you need a number you don't already have, call the
   appropriate tool before answering.
2. You NEVER fabricate figures, transactions, or account details. If information
   is missing or a tool cannot answer the question, say so plainly and state
   your assumption.
3. Currency values are in Indian Rupees ({CURRENCY_SYMBOL}). Always use this symbol.
4. Every recommendation must be grounded in the tool output you were given and
   must include a short "why" — the reasoning behind it.
5. Keep responses clear and structured. Prefer short paragraphs and bullet
   points over long walls of text.
6. Be encouraging but honest — if the customer's financial health is poor,
   say so respectfully and focus on constructive next steps.
7. When the user references something from earlier in the conversation
   (e.g. "my income is X", "my goal is Y"), incorporate that context into
   your reasoning and tool usage.
8. Never provide investment advice that guarantees returns, and never
   recommend specific stocks, funds, or products. Speak in terms of general
   financial principles (emergency funds, savings rate, budget discipline).

## Style
- Address the customer directly and warmly, like a knowledgeable advisor,
  not a robotic report generator.
- Use headings, bold key numbers, and bullet points where helpful.
- Close with a clear, actionable next step when appropriate.
"""


# --------------------------------------------------------------------------- #
# Structured prompts for specific reasoning tasks
# --------------------------------------------------------------------------- #

def expense_analysis_prompt(customer_name: str, month: str) -> str:
    return (
        f"Analyze {customer_name}'s spending for {month}. Use the expense analyzer "
        "and budget analyzer tools to identify overspending categories. Explain "
        "which categories are above the recommended budget share, by how much, "
        "and why that matters for their financial health."
    )


def savings_suggestions_prompt(customer_name: str) -> str:
    return (
        f"Based on {customer_name}'s savings analysis and financial health tools, "
        "suggest concrete ways to increase their monthly savings rate. Ground every "
        "suggestion in the specific category data returned by the tools — do not "
        "give generic advice unrelated to their actual spending pattern."
    )


def financial_health_prompt(customer_name: str) -> str:
    return (
        f"Call the financial health tool for {customer_name} and explain their score "
        "in plain language: what is driving it up or down (savings rate, budget "
        "adherence, expense stability, emergency fund), and what single action "
        "would improve the score the most."
    )


def budget_coaching_prompt(customer_name: str) -> str:
    return (
        f"Act as a budget coach for {customer_name}. Use the budget analyzer tool to "
        "find violations, and the trend analyzer to see if the overspending is a "
        "one-off or a pattern. Give a realistic, prioritized action plan."
    )


def investment_guidance_prompt(customer_name: str) -> str:
    return (
        f"Using {customer_name}'s savings and emergency fund analysis, advise on "
        "whether they are in a position to consider investing surplus savings. "
        "Do not recommend specific products — speak only in terms of readiness "
        "(emergency fund coverage, stable surplus, debt/EMI load)."
    )


def risk_analysis_prompt(customer_name: str) -> str:
    return (
        f"Assess {customer_name}'s financial risk exposure using the emergency fund, "
        "cash flow, and trend analysis tools. Identify any months with negative cash "
        "flow, high expense volatility, or inadequate emergency reserves, and explain "
        "the practical risk this creates."
    )


AI_INSIGHTS_INSTRUCTION = """Generate a structured set of financial insights for this customer using
the recommendation_inputs tool (and any other tool needed) as your ONLY source of numbers.

Return between 4 and 7 insights. For EACH insight, use exactly this structure:

**Observation:** <what the data shows>
**Reason:** <why this is happening / why it matters>
**Recommendation:** <specific, actionable step>
**Expected Benefit:** <concrete, realistic upside of taking the action>

Cover a mix of: top spending categories, savings opportunities, budget violations,
positive habits worth reinforcing, risk areas, and a forward-looking savings estimate.
Do not repeat the same category in more than one insight unless highlighting a
different angle.
"""


REPORT_SUMMARY_INSTRUCTION = """Write a concise 3-4 sentence executive summary of this customer's
current financial month, suitable for the top of a PDF report. Reference the actual
figures from the tool output (income, expense, savings %, health score) and end with
one clear recommendation."""
