"""
Deterministic Financial Engine.

Every numeric computation in this application happens here — and ONLY
here. The LLM never performs arithmetic; it calls the tool wrappers in
``tools.py``, which in turn call this engine, and the LLM's job is
purely to reason over and explain the already-computed numbers.

Design notes
------------
* All public methods return plain JSON-serialisable dicts so they can
  be handed straight back to the LLM as tool results.
* The engine is stateless per call — it is instantiated with the full
  dataset and a customer_id, then queried for whichever month/window
  is needed. This keeps it easy to unit test and reuse from both the
  Streamlit UI and the agent's tool layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    EMERGENCY_FUND_MONTHS,
    EXPENSE_CATEGORIES,
    HEALTH_SCORE_WEIGHTS,
    IDEAL_SAVINGS_PCT,
    RECOMMENDED_BUDGET_PCT,
)

logger = logging.getLogger(__name__)


class CustomerNotFoundError(Exception):
    """Raised when a requested customer_id does not exist in the dataset."""


class NoDataForMonthError(Exception):
    """Raised when a requested month has no row for the customer."""


@dataclass
class FinanceEngine:
    """Deterministic calculator scoped to a single customer's full history."""

    full_df: pd.DataFrame
    customer_id: str
    _customer_df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        subset = self.full_df[self.full_df["customer_id"] == self.customer_id].copy()
        if subset.empty:
            raise CustomerNotFoundError(f"No records found for customer_id={self.customer_id!r}")
        subset = subset.sort_values("month").reset_index(drop=True)
        self._customer_df = subset

    # ------------------------------------------------------------------ #
    # Basic accessors
    # ------------------------------------------------------------------ #
    @property
    def available_months(self) -> list[str]:
        return self._customer_df["month"].tolist()

    @property
    def latest_month(self) -> str:
        return self._customer_df["month"].iloc[-1]

    def _row_for_month(self, month: str | None) -> pd.Series:
        month = month or self.latest_month
        matches = self._customer_df[self._customer_df["month"] == month]
        if matches.empty:
            raise NoDataForMonthError(f"No data for month={month!r}")
        return matches.iloc[0]

    def full_history_df(self) -> pd.DataFrame:
        """Return a copy of the customer's full month-by-month transaction history."""
        return self._customer_df.copy()

    def customer_profile(self) -> dict:
        row = self._customer_df.iloc[-1]
        return {
            "customer_id": self.customer_id,
            "name": row["customer_name"],
            "age": int(row["age"]),
            "occupation": row["occupation"],
        }

    # ------------------------------------------------------------------ #
    # 1. Monthly summary
    # ------------------------------------------------------------------ #
    def monthly_summary(self, month: str | None = None) -> dict:
        row = self._row_for_month(month)
        income = float(row["monthly_income"])
        expense = float(row["total_expense"])
        savings = float(row["savings"])
        savings_pct = round((savings / income) * 100, 2) if income else 0.0
        expense_ratio = round((expense / income) * 100, 2) if income else 0.0

        category_expenses = {cat: float(row[cat]) for cat in EXPENSE_CATEGORIES}
        highest_cat = max(category_expenses, key=category_expenses.get)
        lowest_cat = min(category_expenses, key=category_expenses.get)

        return {
            "month": row["month"],
            "monthly_income": round(income, 2),
            "total_expense": round(expense, 2),
            "net_savings": round(savings, 2),
            "savings_pct": savings_pct,
            "expense_ratio_pct": expense_ratio,
            "category_expenses": {k: round(v, 2) for k, v in category_expenses.items()},
            "highest_expense_category": highest_cat,
            "highest_expense_amount": round(category_expenses[highest_cat], 2),
            "lowest_expense_category": lowest_cat,
            "lowest_expense_amount": round(category_expenses[lowest_cat], 2),
        }

    # ------------------------------------------------------------------ #
    # 2. Category / expense analysis
    # ------------------------------------------------------------------ #
    def expense_analysis(self, month: str | None = None) -> dict:
        summary = self.monthly_summary(month)
        income = summary["monthly_income"]
        category_expenses = summary["category_expenses"]

        category_ratios = {
            cat: round((amt / income) * 100, 2) if income else 0.0
            for cat, amt in category_expenses.items()
        }

        overspending = [
            {
                "category": cat,
                "spent_pct": category_ratios[cat],
                "recommended_pct": round(RECOMMENDED_BUDGET_PCT[cat] * 100, 2),
                "overspend_pct_points": round(
                    category_ratios[cat] - RECOMMENDED_BUDGET_PCT[cat] * 100, 2
                ),
            }
            for cat in EXPENSE_CATEGORIES
            if category_ratios[cat] > RECOMMENDED_BUDGET_PCT[cat] * 100
        ]
        overspending.sort(key=lambda x: x["overspend_pct_points"], reverse=True)

        sorted_categories = sorted(category_expenses.items(), key=lambda kv: kv[1], reverse=True)

        return {
            "month": summary["month"],
            "category_expenses": category_expenses,
            "category_ratios_pct": category_ratios,
            "overspending_categories": overspending,
            "top_3_expense_categories": [
                {"category": c, "amount": a} for c, a in sorted_categories[:3]
            ],
        }

    # ------------------------------------------------------------------ #
    # 3. Savings calculator
    # ------------------------------------------------------------------ #
    def savings_analysis(self, month: str | None = None) -> dict:
        summary = self.monthly_summary(month)
        gap_to_ideal_pct = round(IDEAL_SAVINGS_PCT * 100 - summary["savings_pct"], 2)
        ideal_savings_amount = round(summary["monthly_income"] * IDEAL_SAVINGS_PCT, 2)
        return {
            "month": summary["month"],
            "net_savings": summary["net_savings"],
            "savings_pct": summary["savings_pct"],
            "ideal_savings_pct": round(IDEAL_SAVINGS_PCT * 100, 2),
            "ideal_savings_amount": ideal_savings_amount,
            "gap_to_ideal_pct_points": gap_to_ideal_pct,
            "meets_ideal_savings_target": summary["savings_pct"] >= IDEAL_SAVINGS_PCT * 100,
        }

    def savings_growth(self) -> dict:
        """Month-over-month growth in absolute savings across full history."""
        df = self._customer_df
        savings_series = df["savings"].tolist()
        months = df["month"].tolist()

        growth = []
        for i in range(1, len(savings_series)):
            prev, curr = savings_series[i - 1], savings_series[i]
            change = curr - prev
            pct_change = round((change / prev) * 100, 2) if prev else 0.0
            growth.append(
                {
                    "month": months[i],
                    "savings": round(curr, 2),
                    "change_from_prev_month": round(change, 2),
                    "pct_change_from_prev_month": pct_change,
                }
            )

        avg_monthly_savings = round(float(np.mean(savings_series)), 2)
        return {
            "monthly_savings_history": [
                {"month": m, "savings": round(s, 2)} for m, s in zip(months, savings_series)
            ],
            "growth_trend": growth,
            "average_monthly_savings": avg_monthly_savings,
            "total_savings_over_period": round(float(np.sum(savings_series)), 2),
        }

    # ------------------------------------------------------------------ #
    # 4. Budget analyzer
    # ------------------------------------------------------------------ #
    def budget_analysis(self, month: str | None = None) -> dict:
        analysis = self.expense_analysis(month)
        violations = analysis["overspending_categories"]
        total_budget_pct = sum(RECOMMENDED_BUDGET_PCT.values()) * 100
        total_spent_pct = sum(analysis["category_ratios_pct"].values())
        utilization_pct = round((total_spent_pct / total_budget_pct) * 100, 2) if total_budget_pct else 0.0

        return {
            "month": analysis["month"],
            "budget_violations": violations,
            "violation_count": len(violations),
            "budget_utilization_pct": utilization_pct,
            "within_budget": len(violations) == 0,
        }

    # ------------------------------------------------------------------ #
    # 5. Trend analyzer
    # ------------------------------------------------------------------ #
    def trend_analysis(self, n_months: int | None = None) -> dict:
        df = self._customer_df.tail(n_months) if n_months else self._customer_df
        months = df["month"].tolist()
        expenses = df["total_expense"].tolist()
        incomes = df["monthly_income"].tolist()

        monthly_changes = []
        for i in range(1, len(expenses)):
            change = expenses[i] - expenses[i - 1]
            pct = round((change / expenses[i - 1]) * 100, 2) if expenses[i - 1] else 0.0
            monthly_changes.append(
                {
                    "month": months[i],
                    "expense": round(expenses[i], 2),
                    "change_from_prev_month": round(change, 2),
                    "pct_change_from_prev_month": pct,
                    "direction": "increase" if change > 0 else ("decrease" if change < 0 else "flat"),
                }
            )

        # Category-level trend: which categories rose the most over the window
        category_trend = {}
        for cat in EXPENSE_CATEGORIES:
            series = df[cat].tolist()
            if len(series) >= 2 and series[0] != 0:
                pct_change = round(((series[-1] - series[0]) / series[0]) * 100, 2)
            else:
                pct_change = 0.0
            category_trend[cat] = pct_change

        rising_categories = sorted(
            ({"category": c, "pct_change_over_window": v} for c, v in category_trend.items()),
            key=lambda x: x["pct_change_over_window"],
            reverse=True,
        )[:3]

        return {
            "months_analyzed": months,
            "average_monthly_expense": round(float(np.mean(expenses)), 2),
            "average_monthly_income": round(float(np.mean(incomes)), 2),
            "monthly_expense_trend": monthly_changes,
            "top_rising_categories": rising_categories,
            "expense_volatility_std": round(float(np.std(expenses)), 2),
        }

    # ------------------------------------------------------------------ #
    # 6. Financial Health Score (0-100)
    # ------------------------------------------------------------------ #
    def financial_health_score(self, month: str | None = None) -> dict:
        summary = self.monthly_summary(month)
        budget = self.budget_analysis(month)
        emergency = self.emergency_fund_estimate(month)
        trend = self.trend_analysis(n_months=6)

        # 1) Savings rate score: 0 at 0% savings, 100 at IDEAL_SAVINGS_PCT or above
        savings_score = min(100.0, max(0.0, (summary["savings_pct"] / (IDEAL_SAVINGS_PCT * 100)) * 100))

        # 2) Budget adherence score: 100 - (violation penalty)
        budget_score = max(0.0, 100.0 - budget["violation_count"] * 12.5)

        # 3) Expense stability score: lower volatility relative to mean = higher score
        volatility_ratio = (
            trend["expense_volatility_std"] / trend["average_monthly_expense"]
            if trend["average_monthly_expense"]
            else 0.0
        )
        stability_score = max(0.0, 100.0 - volatility_ratio * 200)

        # 4) Emergency fund score: 100 at EMERGENCY_FUND_MONTHS coverage or above
        fund_score = min(100.0, (emergency["months_covered"] / EMERGENCY_FUND_MONTHS) * 100)

        weighted_score = (
            savings_score * HEALTH_SCORE_WEIGHTS["savings_rate"]
            + budget_score * HEALTH_SCORE_WEIGHTS["budget_adherence"]
            + stability_score * HEALTH_SCORE_WEIGHTS["expense_stability"]
            + fund_score * HEALTH_SCORE_WEIGHTS["emergency_fund"]
        )
        final_score = round(min(100.0, max(0.0, weighted_score)), 1)

        if final_score >= 80:
            rating = "Excellent"
        elif final_score >= 60:
            rating = "Good"
        elif final_score >= 40:
            rating = "Fair"
        else:
            rating = "Needs Attention"

        return {
            "month": summary["month"],
            "financial_health_score": final_score,
            "rating": rating,
            "component_scores": {
                "savings_rate_score": round(savings_score, 1),
                "budget_adherence_score": round(budget_score, 1),
                "expense_stability_score": round(stability_score, 1),
                "emergency_fund_score": round(fund_score, 1),
            },
            "weights_used": HEALTH_SCORE_WEIGHTS,
        }

    # ------------------------------------------------------------------ #
    # 7. Emergency fund estimate
    # ------------------------------------------------------------------ #
    def emergency_fund_estimate(self, month: str | None = None) -> dict:
        summary = self.monthly_summary(month)
        growth = self.savings_growth()
        avg_expense = self.trend_analysis()["average_monthly_expense"]
        total_savings = growth["total_savings_over_period"]

        months_covered = round(total_savings / avg_expense, 2) if avg_expense else 0.0
        recommended_fund = round(avg_expense * EMERGENCY_FUND_MONTHS, 2)
        shortfall = round(max(0.0, recommended_fund - total_savings), 2)

        return {
            "month": summary["month"],
            "estimated_liquid_savings": round(total_savings, 2),
            "average_monthly_expense": avg_expense,
            "months_covered": months_covered,
            "recommended_months_coverage": EMERGENCY_FUND_MONTHS,
            "recommended_fund_amount": recommended_fund,
            "shortfall": shortfall,
            "is_adequately_covered": months_covered >= EMERGENCY_FUND_MONTHS,
        }

    # ------------------------------------------------------------------ #
    # 8. Cash flow analysis
    # ------------------------------------------------------------------ #
    def cash_flow_analysis(self) -> dict:
        df = self._customer_df
        months = df["month"].tolist()
        inflow = df["monthly_income"].tolist()
        outflow = df["total_expense"].tolist()
        net_flow = [round(i - o, 2) for i, o in zip(inflow, outflow)]

        positive_months = sum(1 for n in net_flow if n > 0)
        negative_months = sum(1 for n in net_flow if n < 0)

        return {
            "months": months,
            "inflow": [round(x, 2) for x in inflow],
            "outflow": [round(x, 2) for x in outflow],
            "net_cash_flow": net_flow,
            "positive_cash_flow_months": positive_months,
            "negative_cash_flow_months": negative_months,
            "average_net_cash_flow": round(float(np.mean(net_flow)), 2),
        }

    # ------------------------------------------------------------------ #
    # 9. Recommendation inputs (deterministic facts; LLM phrases them)
    # ------------------------------------------------------------------ #
    def recommendation_inputs(self, month: str | None = None) -> dict:
        """
        Bundle every deterministic fact the LLM needs to generate grounded,
        non-hallucinated recommendations. This tool performs NO reasoning —
        it simply gathers the other calculations into one payload.
        """
        return {
            "summary": self.monthly_summary(month),
            "expense_analysis": self.expense_analysis(month),
            "savings_analysis": self.savings_analysis(month),
            "budget_analysis": self.budget_analysis(month),
            "health_score": self.financial_health_score(month),
            "emergency_fund": self.emergency_fund_estimate(month),
        }

    # ------------------------------------------------------------------ #
    # 10. Affordability check (what-if spending)
    # ------------------------------------------------------------------ #
    def affordability_check(self, amount: float, month: str | None = None) -> dict:
        """Deterministic check for 'Can I afford X?' style questions."""
        summary = self.monthly_summary(month)
        emergency = self.emergency_fund_estimate(month)
        net_savings = summary["net_savings"]

        affordable_from_monthly_savings = amount <= net_savings
        affordable_without_breaking_emergency_fund = (
            emergency["estimated_liquid_savings"] - amount
        ) >= emergency["recommended_fund_amount"]

        months_to_save = (
            round(amount / net_savings, 1) if net_savings > 0 else None
        )

        return {
            "requested_amount": round(amount, 2),
            "current_month_net_savings": net_savings,
            "affordable_from_this_months_savings": affordable_from_monthly_savings,
            "affordable_without_touching_emergency_fund": affordable_without_breaking_emergency_fund,
            "estimated_months_to_save_up": months_to_save,
        }

    # ------------------------------------------------------------------ #
    # 11. Month comparison
    # ------------------------------------------------------------------ #
    def compare_months(self, month_a: str, month_b: str) -> dict:
        a = self.monthly_summary(month_a)
        b = self.monthly_summary(month_b)

        category_diff = {
            cat: round(b["category_expenses"][cat] - a["category_expenses"][cat], 2)
            for cat in EXPENSE_CATEGORIES
        }

        return {
            "month_a": a["month"],
            "month_b": b["month"],
            "income_change": round(b["monthly_income"] - a["monthly_income"], 2),
            "expense_change": round(b["total_expense"] - a["total_expense"], 2),
            "savings_change": round(b["net_savings"] - a["net_savings"], 2),
            "savings_pct_change_points": round(b["savings_pct"] - a["savings_pct"], 2),
            "category_changes": category_diff,
        }
