"""
Synthetic customer & transaction data generator.

Produces a realistic 12-month expense history for 10 banking customers
spanning different income brackets, occupations and spending habits.
The output is written to ``data/customers.csv`` and is the single
source of truth consumed by ``finance_engine.py``.

Run directly to (re)generate the dataset:

    python sample_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CUSTOMERS_CSV, EXPENSE_CATEGORIES, MONTHS_OF_HISTORY

RNG_SEED = 42

# --------------------------------------------------------------------------- #
# Customer profiles
# --------------------------------------------------------------------------- #
CUSTOMER_PROFILES = [
    {"customer_id": "CUST001", "name": "Aarav Sharma", "age": 29, "occupation": "Software Engineer", "income": 95000, "savings_habit": "good"},
    {"customer_id": "CUST002", "name": "Priya Nair", "age": 34, "occupation": "Marketing Manager", "income": 110000, "savings_habit": "average"},
    {"customer_id": "CUST003", "name": "Rohan Verma", "age": 26, "occupation": "Data Analyst", "income": 65000, "savings_habit": "poor"},
    {"customer_id": "CUST004", "name": "Sneha Iyer", "age": 41, "occupation": "Doctor", "income": 180000, "savings_habit": "good"},
    {"customer_id": "CUST005", "name": "Vikram Singh", "age": 38, "occupation": "Business Owner", "income": 150000, "savings_habit": "average"},
    {"customer_id": "CUST006", "name": "Ananya Reddy", "age": 24, "occupation": "Graphic Designer", "income": 48000, "savings_habit": "poor"},
    {"customer_id": "CUST007", "name": "Karan Mehta", "age": 45, "occupation": "Bank Manager", "income": 135000, "savings_habit": "good"},
    {"customer_id": "CUST008", "name": "Divya Krishnan", "age": 31, "occupation": "Teacher", "income": 58000, "savings_habit": "average"},
    {"customer_id": "CUST009", "name": "Aditya Rao", "age": 27, "occupation": "Civil Engineer", "income": 72000, "savings_habit": "average"},
    {"customer_id": "CUST010", "name": "Ishita Bansal", "age": 36, "occupation": "Product Manager", "income": 145000, "savings_habit": "good"},
]

# Baseline share of income each category tends to consume, keyed by habit tier.
# Values are (mean_pct, std_pct) of monthly income.
CATEGORY_BASELINE = {
    "good": {
        "Food": (0.09, 0.015), "Rent": (0.22, 0.02), "Shopping": (0.05, 0.015),
        "Travel": (0.03, 0.012), "EMI": (0.10, 0.03), "Utilities": (0.04, 0.008),
        "Entertainment": (0.03, 0.01), "Healthcare": (0.03, 0.012), "Insurance": (0.04, 0.008),
        "Education": (0.02, 0.01), "Investments": (0.15, 0.03), "Miscellaneous": (0.02, 0.008),
    },
    "average": {
        "Food": (0.12, 0.02), "Rent": (0.27, 0.02), "Shopping": (0.08, 0.02),
        "Travel": (0.04, 0.015), "EMI": (0.13, 0.03), "Utilities": (0.05, 0.01),
        "Entertainment": (0.04, 0.012), "Healthcare": (0.03, 0.012), "Insurance": (0.04, 0.008),
        "Education": (0.03, 0.012), "Investments": (0.08, 0.02), "Miscellaneous": (0.03, 0.01),
    },
    "poor": {
        "Food": (0.15, 0.025), "Rent": (0.30, 0.02), "Shopping": (0.12, 0.03),
        "Travel": (0.06, 0.02), "EMI": (0.16, 0.03), "Utilities": (0.05, 0.01),
        "Entertainment": (0.06, 0.015), "Healthcare": (0.04, 0.015), "Insurance": (0.03, 0.008),
        "Education": (0.03, 0.012), "Investments": (0.03, 0.01), "Miscellaneous": (0.05, 0.015),
    },
}

# Seasonal multipliers applied to specific categories in specific months
# (1 = January ... 12 = December). Captures festive/holiday spending spikes.
SEASONAL_BUMPS = {
    10: {"Shopping": 1.35, "Entertainment": 1.2},   # Festive season
    11: {"Shopping": 1.25, "Travel": 1.15},
    12: {"Travel": 1.4, "Entertainment": 1.3, "Shopping": 1.2},  # Year-end holidays
    4: {"Education": 1.3},                           # New academic year
    6: {"Healthcare": 1.2},                          # Monsoon illness bump
}


def _generate_month_labels(n_months: int = MONTHS_OF_HISTORY) -> list[str]:
    """Return the last n_months as 'YYYY-MM' strings ending at the current month."""
    end = pd.Timestamp.today().normalize().replace(day=1)
    months = pd.date_range(end=end, periods=n_months, freq="MS")
    return [m.strftime("%Y-%m") for m in months]


def generate_customers_dataframe(seed: int = RNG_SEED) -> pd.DataFrame:
    """Build the full customer x month transaction table."""
    rng = np.random.default_rng(seed)
    month_labels = _generate_month_labels()
    rows: list[dict] = []

    for profile in CUSTOMER_PROFILES:
        habit = profile["savings_habit"]
        baseline = CATEGORY_BASELINE[habit]
        income = profile["income"]

        # Slight month-to-month income drift (e.g. bonus, appraisal) for realism.
        income_drift = rng.normal(loc=1.0, scale=0.01, size=len(month_labels))
        income_drift[-1] *= 1.0  # no forced change on latest month

        for m_idx, month in enumerate(month_labels):
            month_num = int(month.split("-")[1])
            monthly_income = round(income * income_drift[m_idx], 2)

            category_values: dict[str, float] = {}
            for category in EXPENSE_CATEGORIES:
                mean_pct, std_pct = baseline[category]
                seasonal_mult = SEASONAL_BUMPS.get(month_num, {}).get(category, 1.0)
                pct = max(rng.normal(mean_pct, std_pct), 0.005) * seasonal_mult
                category_values[category] = round(monthly_income * pct, 2)

            total_expense = round(sum(category_values.values()), 2)
            savings = round(monthly_income - total_expense, 2)
            savings_pct = round((savings / monthly_income) * 100, 2) if monthly_income else 0.0

            row = {
                "customer_id": profile["customer_id"],
                "customer_name": profile["name"],
                "age": profile["age"],
                "occupation": profile["occupation"],
                "month": month,
                "monthly_income": monthly_income,
                **category_values,
                "total_expense": total_expense,
                "savings": savings,
                "savings_pct": savings_pct,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def save_customers_csv(path=CUSTOMERS_CSV) -> pd.DataFrame:
    """Generate the dataset and persist it to disk."""
    df = generate_customers_dataframe()
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = save_customers_csv()
    print(f"Generated {len(df)} rows for {df['customer_id'].nunique()} customers "
          f"across {df['month'].nunique()} months -> {CUSTOMERS_CSV}")
