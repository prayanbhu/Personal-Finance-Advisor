"""
Shared utility helpers: data loading, formatting, and report generation.

Kept free of Streamlit/LLM imports so it can be reused by the CLI,
tests, or a future API layer without pulling in the whole stack.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from config import CURRENCY_SYMBOL, CUSTOMERS_CSV

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_customer_data(path=CUSTOMERS_CSV) -> pd.DataFrame:
    """Load the full customer transaction dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Customer dataset not found at {path}. Run `python sample_data.py` first."
        )
    df = pd.read_csv(path)
    return df


def get_customer_list(df: pd.DataFrame) -> list[dict]:
    """Return a de-duplicated list of {customer_id, name} for UI selection."""
    unique = df.drop_duplicates(subset="customer_id")[["customer_id", "customer_name"]]
    return [
        {"customer_id": r.customer_id, "customer_name": r.customer_name}
        for r in unique.itertuples()
    ]


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def format_currency(amount: float) -> str:
    """Format a number as Indian-style currency, e.g. 125000 -> ₹1,25,000.00"""
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return f"{CURRENCY_SYMBOL}0.00"

    negative = amount < 0
    amount = abs(amount)
    whole, frac = f"{amount:,.2f}".split(".")
    whole = whole.replace(",", "")

    if len(whole) > 3:
        last3 = whole[-3:]
        rest = whole[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        formatted = ",".join(groups) + "," + last3
    else:
        formatted = whole

    sign = "-" if negative else ""
    return f"{sign}{CURRENCY_SYMBOL}{formatted}.{frac}"


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
def build_csv_report(report_data: dict) -> bytes:
    """Flatten a financial report dict into a downloadable CSV byte string."""
    rows = []

    def _flatten(prefix: str, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _flatten(f"{prefix}[{i}]", v)
        else:
            rows.append({"metric": prefix, "value": obj})

    _flatten("", report_data)
    df = pd.DataFrame(rows)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _sanitize_for_pdf(text: str) -> str:
    """
    fpdf2's core (non-embedded) fonts only support the latin-1 range, so the
    Rupee sign (and any other AI-generated unicode) must be substituted
    before it reaches the PDF, not just replaced with '?' on encode failure.
    """
    text = text.replace(CURRENCY_SYMBOL, "Rs. ")
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf_report(
    customer_name: str,
    month: str,
    summary: dict,
    health: dict,
    recommendations: list[str],
    ai_summary: str = "",
) -> bytes:
    """Generate a downloadable one-page PDF financial summary report."""
    NEXT_LINE = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Personal Finance Advisor - Monthly Report", **NEXT_LINE)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, _sanitize_for_pdf(f"Customer: {customer_name}"), **NEXT_LINE)
    pdf.cell(0, 8, f"Month: {month}", **NEXT_LINE)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", **NEXT_LINE)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Financial Summary", **NEXT_LINE)
    pdf.set_font("Helvetica", "", 11)
    summary_lines = [
        f"Monthly Income: {format_currency(summary.get('monthly_income', 0))}",
        f"Total Expense: {format_currency(summary.get('total_expense', 0))}",
        f"Net Savings: {format_currency(summary.get('net_savings', 0))}",
        f"Savings %: {format_pct(summary.get('savings_pct', 0))}",
        f"Expense Ratio: {format_pct(summary.get('expense_ratio_pct', 0))}",
        f"Highest Expense Category: {summary.get('highest_expense_category', '-')}",
    ]
    for line in summary_lines:
        pdf.cell(0, 7, _sanitize_for_pdf(line), **NEXT_LINE)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Financial Health Score", **NEXT_LINE)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(
        0, 7,
        f"Score: {health.get('financial_health_score', 0)}/100  ({health.get('rating', '-')})",
        **NEXT_LINE,
    )
    pdf.ln(3)

    if ai_summary:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "AI Insights Summary", **NEXT_LINE)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _sanitize_for_pdf(ai_summary), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    if recommendations:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Top Recommendations", **NEXT_LINE)
        pdf.set_font("Helvetica", "", 10)
        for i, rec in enumerate(recommendations, start=1):
            pdf.multi_cell(0, 6, _sanitize_for_pdf(f"{i}. {rec}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    output = pdf.output()
    return bytes(output)
