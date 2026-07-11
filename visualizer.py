"""
Visualization layer — interactive Plotly charts for the Streamlit dashboard.

Color usage follows a magnitude-vs-identity discipline:
* Charts comparing magnitude across many categories (bar, heatmap, treemap)
  use a single sequential hue, shaded light->dark by value.
* Charts where category *identity* matters (pie, stacked bar) use the fixed
  categorical palette for the top slots and fold the long tail into "Other"
  so no chart ever exceeds ~8 color tokens.
* Gauges use the fixed status palette (good/warning/serious/critical) since
  they represent a state, not a series.

All functions accept a `dark_mode` flag and theme themselves accordingly
rather than relying on Plotly's default templates, so the dashboard's theme
toggle affects charts too.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import CURRENCY_SYMBOL, EXPENSE_CATEGORIES

# --------------------------------------------------------------------------- #
# Palette (validated reference palette — see dataviz skill)
# --------------------------------------------------------------------------- #
CATEGORICAL = {
    "light": ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"],
    "dark":  ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"],
}
SEQUENTIAL_RAMP = {
    "light": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "dark":  ["#184f95", "#1c5cab", "#256abf", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"],
}
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
OTHER_GRAY = {"light": "#898781", "dark": "#898781"}


def _theme(dark_mode: bool) -> dict:
    mode = "dark" if dark_mode else "light"
    return {
        "surface": "#1a1a19" if dark_mode else "#fcfcfb",
        "primary_ink": "#ffffff" if dark_mode else "#0b0b0b",
        "secondary_ink": "#c3c2b7" if dark_mode else "#52514e",
        "muted_ink": "#898781",
        "gridline": "#2c2c2a" if dark_mode else "#e1e0d9",
        "categorical": CATEGORICAL[mode],
        "sequential": SEQUENTIAL_RAMP[mode],
        "other_gray": OTHER_GRAY[mode],
    }


def _base_layout(fig: go.Figure, theme: dict, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=theme["primary_ink"]), x=0.02, xanchor="left"),
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=theme["secondary_ink"]),
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=theme["secondary_ink"])),
        hoverlabel=dict(bgcolor=theme["surface"], font_color=theme["primary_ink"]),
    )
    fig.update_xaxes(gridcolor=theme["gridline"], linecolor=theme["gridline"], zeroline=False,
                      tickfont=dict(color=theme["muted_ink"]))
    fig.update_yaxes(gridcolor=theme["gridline"], linecolor=theme["gridline"], zeroline=False,
                      tickfont=dict(color=theme["muted_ink"]))
    return fig


def _fold_into_other(category_expenses: dict, max_slots: int = 7) -> dict:
    """Sort descending and fold the long tail into 'Other' to respect the
    categorical token ceiling (<=8 including Other)."""
    items = sorted(category_expenses.items(), key=lambda kv: kv[1], reverse=True)
    if len(items) <= max_slots:
        return dict(items)
    head = dict(items[:max_slots])
    tail_sum = sum(v for _, v in items[max_slots:])
    if tail_sum > 0:
        head["Other"] = tail_sum
    return head


# --------------------------------------------------------------------------- #
# 1. Pie / Donut — Expense Distribution
# --------------------------------------------------------------------------- #
def expense_distribution_pie(category_expenses: dict, dark_mode: bool = False) -> go.Figure:
    theme = _theme(dark_mode)
    folded = _fold_into_other(category_expenses)
    labels = list(folded.keys())
    values = list(folded.values())
    colors = theme["categorical"][: len(labels)]
    if "Other" in labels:
        colors[labels.index("Other")] = theme["other_gray"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color=theme["surface"], width=2)),
                textinfo="label+percent",
                textfont=dict(color=theme["primary_ink"], size=12),
                hovertemplate=f"%{{label}}: {CURRENCY_SYMBOL}%{{value:,.0f}} (%{{percent}})<extra></extra>",
            )
        ]
    )
    return _base_layout(fig, theme, "Expense Distribution")


# --------------------------------------------------------------------------- #
# 2. Bar Chart — Expense Categories (magnitude, sequential shading)
# --------------------------------------------------------------------------- #
def expense_categories_bar(category_expenses: dict, dark_mode: bool = False) -> go.Figure:
    theme = _theme(dark_mode)
    items = sorted(category_expenses.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=values, colorscale=[[i / 6, c] for i, c in enumerate(theme["sequential"])],
                            line=dict(width=0)),
                text=[f"{CURRENCY_SYMBOL}{v:,.0f}" for v in values],
                textposition="outside",
                textfont=dict(color=theme["secondary_ink"]),
                hovertemplate=f"%{{y}}: {CURRENCY_SYMBOL}%{{x:,.0f}}<extra></extra>",
            )
        ]
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(bargap=0.3)
    return _base_layout(fig, theme, "Expense Categories", height=420)


# --------------------------------------------------------------------------- #
# 3. Line Chart — Monthly Expenses
# --------------------------------------------------------------------------- #
def monthly_expense_line(months: list[str], expenses: list[float], dark_mode: bool = False) -> go.Figure:
    theme = _theme(dark_mode)
    color = theme["categorical"][0]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=months, y=expenses, mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color, line=dict(color=theme["surface"], width=2)),
                fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.1),
                hovertemplate=f"%{{x}}: {CURRENCY_SYMBOL}%{{y:,.0f}}<extra></extra>",
            )
        ]
    )
    return _base_layout(fig, theme, "Monthly Expense Trend")


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------- #
# 4. Line Chart — Savings Trend
# --------------------------------------------------------------------------- #
def savings_trend_line(months: list[str], savings: list[float], dark_mode: bool = False) -> go.Figure:
    theme = _theme(dark_mode)
    color = theme["categorical"][1]  # aqua — visually distinct from expense line

    fig = go.Figure(
        data=[
            go.Scatter(
                x=months, y=savings, mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color, line=dict(color=theme["surface"], width=2)),
                fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.1),
                hovertemplate=f"%{{x}}: {CURRENCY_SYMBOL}%{{y:,.0f}}<extra></extra>",
            )
        ]
    )
    fig.add_hline(y=0, line_color=theme["gridline"], line_width=1)
    return _base_layout(fig, theme, "Savings Trend")


# --------------------------------------------------------------------------- #
# 5. Stacked Bar — Category Comparison across months
# --------------------------------------------------------------------------- #
def category_comparison_stacked_bar(df: pd.DataFrame, dark_mode: bool = False, max_categories: int = 5) -> go.Figure:
    """df must contain a 'month' column plus the EXPENSE_CATEGORIES columns."""
    theme = _theme(dark_mode)

    totals = df[EXPENSE_CATEGORIES].sum().sort_values(ascending=False)
    top_categories = totals.index[:max_categories].tolist()
    other_categories = totals.index[max_categories:].tolist()

    fig = go.Figure()
    colors = theme["categorical"]
    for i, cat in enumerate(top_categories):
        fig.add_trace(
            go.Bar(
                x=df["month"], y=df[cat], name=cat,
                marker=dict(color=colors[i % len(colors)]),
                hovertemplate=f"{cat}: {CURRENCY_SYMBOL}%{{y:,.0f}}<extra></extra>",
            )
        )
    if other_categories:
        other_sum = df[other_categories].sum(axis=1)
        fig.add_trace(
            go.Bar(
                x=df["month"], y=other_sum, name="Other",
                marker=dict(color=theme["other_gray"]),
                hovertemplate=f"Other: {CURRENCY_SYMBOL}%{{y:,.0f}}<extra></extra>",
            )
        )

    fig.update_layout(barmode="stack", bargap=0.25)
    return _base_layout(fig, theme, "Category Comparison Across Months", height=420)


# --------------------------------------------------------------------------- #
# 6 & 7. Gauge Charts — Financial Health Score / Savings %
# --------------------------------------------------------------------------- #
def _status_band_color(value: float, bands: list[tuple[float, str]]) -> str:
    for threshold, color in bands:
        if value < threshold:
            return color
    return bands[-1][1]


def financial_health_gauge(score: float, dark_mode: bool = False) -> go.Figure:
    theme = _theme(dark_mode)
    bar_color = _status_band_color(score, [(40, STATUS["critical"]), (60, STATUS["serious"]), (80, STATUS["warning"]), (101, STATUS["good"])])

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(suffix=" / 100", font=dict(color=theme["primary_ink"], size=32)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=theme["muted_ink"], tickfont=dict(color=theme["muted_ink"])),
                bar=dict(color=bar_color, thickness=0.35),
                bgcolor=theme["surface"],
                borderwidth=0,
                steps=[
                    {"range": [0, 40], "color": _hex_to_rgba(STATUS["critical"], 0.15)},
                    {"range": [40, 60], "color": _hex_to_rgba(STATUS["serious"], 0.15)},
                    {"range": [60, 80], "color": _hex_to_rgba(STATUS["warning"], 0.15)},
                    {"range": [80, 100], "color": _hex_to_rgba(STATUS["good"], 0.15)},
                ],
            ),
        )
    )
    return _base_layout(fig, theme, "Financial Health Score", height=300)


def savings_pct_gauge(savings_pct: float, ideal_pct: float = 20.0, dark_mode: bool = False) -> go.Figure:
    theme = _theme(dark_mode)
    max_range = max(40.0, ideal_pct * 1.5, savings_pct * 1.2)
    bar_color = STATUS["good"] if savings_pct >= ideal_pct else (
        STATUS["warning"] if savings_pct >= ideal_pct * 0.5 else STATUS["critical"]
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=savings_pct,
            number=dict(suffix="%", font=dict(color=theme["primary_ink"], size=32)),
            gauge=dict(
                axis=dict(range=[0, max_range], tickcolor=theme["muted_ink"], tickfont=dict(color=theme["muted_ink"])),
                bar=dict(color=bar_color, thickness=0.35),
                bgcolor=theme["surface"],
                borderwidth=0,
                threshold=dict(line=dict(color=theme["primary_ink"], width=2), thickness=0.75, value=ideal_pct),
            ),
        )
    )
    return _base_layout(fig, theme, "Savings % (target marker = ideal)", height=300)


# --------------------------------------------------------------------------- #
# 8. Heatmap — Monthly Spending (category x month, sequential)
# --------------------------------------------------------------------------- #
def monthly_spending_heatmap(df: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    theme = _theme(dark_mode)
    matrix = df.set_index("month")[EXPENSE_CATEGORIES].T

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale=[[i / 6, c] for i, c in enumerate(theme["sequential"])],
            colorbar=dict(title=CURRENCY_SYMBOL, tickfont=dict(color=theme["muted_ink"])),
            xgap=2, ygap=2,
            hovertemplate=f"%{{y}} · %{{x}}: {CURRENCY_SYMBOL}%{{z:,.0f}}<extra></extra>",
        )
    )
    return _base_layout(fig, theme, "Monthly Spending Heatmap", height=420)


# --------------------------------------------------------------------------- #
# 9. Treemap — Expense Breakdown (magnitude, sequential)
# --------------------------------------------------------------------------- #
def expense_treemap(category_expenses: dict, dark_mode: bool = False) -> go.Figure:
    theme = _theme(dark_mode)
    labels = list(category_expenses.keys())
    values = list(category_expenses.values())

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=[""] * len(labels),
            values=values,
            marker=dict(
                colors=values,
                colorscale=[[i / 6, c] for i, c in enumerate(theme["sequential"])],
                line=dict(width=2, color=theme["surface"]),
            ),
            texttemplate=f"%{{label}}<br>{CURRENCY_SYMBOL}%{{value:,.0f}}",
            textfont=dict(color=theme["primary_ink"]),
            hovertemplate=f"%{{label}}: {CURRENCY_SYMBOL}%{{value:,.0f}}<extra></extra>",
        )
    )
    return _base_layout(fig, theme, "Expense Breakdown", height=420)
