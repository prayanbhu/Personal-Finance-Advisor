"""
Central configuration for the Personal Finance AI Agent.

All environment variables, file paths, model settings, and tunable
business constants live here so the rest of the application never
reaches into `os.environ` directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

CUSTOMERS_CSV = DATA_DIR / "customers.csv"

for _dir in (DATA_DIR, REPORTS_DIR, ASSETS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Groq / LLM settings
# --------------------------------------------------------------------------- #
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))

# --------------------------------------------------------------------------- #
# Business constants (used by the deterministic finance engine)
# --------------------------------------------------------------------------- #
EXPENSE_CATEGORIES: list[str] = [
    "Food",
    "Rent",
    "Shopping",
    "Travel",
    "EMI",
    "Utilities",
    "Entertainment",
    "Healthcare",
    "Insurance",
    "Education",
    "Investments",
    "Miscellaneous",
]

# Recommended budget ceiling per category, expressed as a % of monthly income.
# Used by the Budget Analyzer tool to flag overspending.
RECOMMENDED_BUDGET_PCT: dict[str, float] = {
    "Food": 0.12,
    "Rent": 0.30,
    "Shopping": 0.08,
    "Travel": 0.05,
    "EMI": 0.15,
    "Utilities": 0.05,
    "Entertainment": 0.05,
    "Healthcare": 0.05,
    "Insurance": 0.05,
    "Education": 0.05,
    "Investments": 0.10,
    "Miscellaneous": 0.03,
}

IDEAL_SAVINGS_PCT = 0.20          # 20% savings rate is considered "healthy"
EMERGENCY_FUND_MONTHS = 6         # Recommended emergency fund coverage
MONTHS_OF_HISTORY = 12

# Financial Health Score weightings (must sum to 1.0)
HEALTH_SCORE_WEIGHTS = {
    "savings_rate": 0.35,
    "budget_adherence": 0.25,
    "expense_stability": 0.20,
    "emergency_fund": 0.20,
}

CURRENCY_SYMBOL = "₹"  # INR

# --------------------------------------------------------------------------- #
# App metadata
# --------------------------------------------------------------------------- #
APP_TITLE = "Personal Finance AI Advisor"
APP_ICON = "💰"
