"""
Agent tool layer.

Wraps ``FinanceEngine`` methods as LangChain ``@tool`` functions so the
LLM can decide, at conversation time, which calculation it needs and
invoke it with structured arguments. Every tool here is a thin,
side-effect-free adapter — all real computation lives in
``finance_engine.py``.

Tools are built per-session via ``build_tool_registry`` because each
tool must be scoped to the currently selected customer's data without
forcing the LLM to pass a customer_id on every call (the customer is
already fixed by the UI, not by the model).
"""

from __future__ import annotations

import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from finance_engine import FinanceEngine

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Argument schemas
# --------------------------------------------------------------------------- #
class MonthArg(BaseModel):
    month: str | None = Field(
        default=None,
        description="Target month in 'YYYY-MM' format. Omit to use the most recent available month.",
    )


class TrendArg(BaseModel):
    n_months: int | None = Field(
        default=6,
        description="Number of most recent months to analyze for the trend. Defaults to 6.",
    )


class AffordabilityArg(BaseModel):
    amount: float = Field(description="The rupee amount the customer wants to spend.")
    month: str | None = Field(default=None, description="Month to check against, defaults to latest.")


class CompareMonthsArg(BaseModel):
    month_a: str = Field(description="Earlier month in 'YYYY-MM' format.")
    month_b: str = Field(description="Later month in 'YYYY-MM' format, to compare against month_a.")


class NoArgs(BaseModel):
    pass


# --------------------------------------------------------------------------- #
# Tool registry factory
# --------------------------------------------------------------------------- #
def build_tool_registry(engine: FinanceEngine) -> list[StructuredTool]:
    """
    Build the full list of LangChain tools bound to a specific customer's
    FinanceEngine instance. Called once per selected customer/session.
    """

    def expense_analyzer(month: str | None = None) -> dict:
        return engine.expense_analysis(month)

    def savings_calculator(month: str | None = None) -> dict:
        return engine.savings_analysis(month)

    def savings_growth_tool(_: None = None) -> dict:
        return engine.savings_growth()

    def budget_analyzer(month: str | None = None) -> dict:
        return engine.budget_analysis(month)

    def trend_analyzer(n_months: int | None = 6) -> dict:
        return engine.trend_analysis(n_months)

    def financial_health_tool(month: str | None = None) -> dict:
        return engine.financial_health_score(month)

    def emergency_fund_tool(month: str | None = None) -> dict:
        return engine.emergency_fund_estimate(month)

    def cash_flow_tool(_: None = None) -> dict:
        return engine.cash_flow_analysis()

    def recommendation_data_tool(month: str | None = None) -> dict:
        """Bundles all deterministic metrics needed to ground recommendations."""
        return engine.recommendation_inputs(month)

    def monthly_summary_tool(month: str | None = None) -> dict:
        return engine.monthly_summary(month)

    def affordability_tool(amount: float, month: str | None = None) -> dict:
        return engine.affordability_check(amount, month)

    def compare_months_tool(month_a: str, month_b: str) -> dict:
        return engine.compare_months(month_a, month_b)

    def customer_profile_tool(_: None = None) -> dict:
        profile = engine.customer_profile()
        profile["available_months"] = engine.available_months
        profile["latest_month"] = engine.latest_month
        return profile

    tools = [
        StructuredTool.from_function(
            func=expense_analyzer,
            name="expense_analyzer",
            description=(
                "Analyze category-wise expenses for a given month. Returns each "
                "category's spend, its % of income, and which categories exceed "
                "recommended budget shares. Use for questions about where money "
                "was spent or which categories are overspent."
            ),
            args_schema=MonthArg,
        ),
        StructuredTool.from_function(
            func=savings_calculator,
            name="savings_calculator",
            description=(
                "Calculate net savings and savings percentage for a given month, "
                "and compare against the ideal 20% savings target. Use for questions "
                "about how much the customer is saving or should be saving."
            ),
            args_schema=MonthArg,
        ),
        StructuredTool.from_function(
            func=savings_growth_tool,
            name="savings_growth",
            description=(
                "Return the full month-by-month savings history and growth trend "
                "across all available months. Use for 'how has my savings changed "
                "over time' style questions."
            ),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            func=budget_analyzer,
            name="budget_analyzer",
            description=(
                "Check for budget violations in a given month — categories where "
                "spending exceeds the recommended limit — and compute overall "
                "budget utilization percentage."
            ),
            args_schema=MonthArg,
        ),
        StructuredTool.from_function(
            func=trend_analyzer,
            name="trend_analyzer",
            description=(
                "Analyze expense trends over the last N months: month-over-month "
                "increases/decreases, which categories are rising fastest, and "
                "expense volatility. Use for 'am I spending more over time' questions."
            ),
            args_schema=TrendArg,
        ),
        StructuredTool.from_function(
            func=financial_health_tool,
            name="financial_health_score",
            description=(
                "Compute the customer's Financial Health Score (0-100) for a given "
                "month, broken down into savings rate, budget adherence, expense "
                "stability, and emergency fund components. Use for overall financial "
                "wellness questions."
            ),
            args_schema=MonthArg,
        ),
        StructuredTool.from_function(
            func=emergency_fund_tool,
            name="emergency_fund_estimate",
            description=(
                "Estimate the customer's emergency fund coverage in months, versus "
                "the recommended 6-month target, based on accumulated savings."
            ),
            args_schema=MonthArg,
        ),
        StructuredTool.from_function(
            func=cash_flow_tool,
            name="cash_flow_analysis",
            description=(
                "Return full-history monthly cash inflow, outflow, and net cash flow, "
                "including counts of positive vs negative cash flow months."
            ),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            func=recommendation_data_tool,
            name="recommendation_inputs",
            description=(
                "Fetch a bundled payload of ALL deterministic metrics (summary, "
                "expense analysis, savings, budget, health score, emergency fund) "
                "for a given month. ALWAYS call this before generating financial "
                "recommendations or AI insights, so every claim is grounded in data."
            ),
            args_schema=MonthArg,
        ),
        StructuredTool.from_function(
            func=monthly_summary_tool,
            name="monthly_summary",
            description=(
                "Get the core monthly summary: income, total expense, net savings, "
                "savings %, expense ratio, and highest/lowest expense categories."
            ),
            args_schema=MonthArg,
        ),
        StructuredTool.from_function(
            func=affordability_tool,
            name="affordability_check",
            description=(
                "Check whether the customer can afford a given rupee amount right "
                "now, based on this month's savings and emergency fund status. Use "
                "for 'can I afford X / buy X / go on vacation' style questions."
            ),
            args_schema=AffordabilityArg,
        ),
        StructuredTool.from_function(
            func=compare_months_tool,
            name="compare_months",
            description=(
                "Compare two specific months' income, expenses, savings, and "
                "category-level changes. Use for 'compare this month with last "
                "month' style questions."
            ),
            args_schema=CompareMonthsArg,
        ),
        StructuredTool.from_function(
            func=customer_profile_tool,
            name="customer_profile",
            description=(
                "Get the customer's profile (name, age, occupation) plus the list "
                "of months available in their transaction history. Use this if you "
                "need to know what months of data exist."
            ),
            args_schema=NoArgs,
        ),
    ]
    return tools
