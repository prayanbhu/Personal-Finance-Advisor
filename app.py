"""
Personal Finance AI Advisor — Streamlit application entry point.

Architecture
------------
    Streamlit UI (this file)
        -> FinanceAgent (agent.py)            [only for AI-powered tabs]
            -> Groq Llama model (llm.py)
            -> Tool calling (tools.py)
                -> FinanceEngine (finance_engine.py)   [deterministic math]
                    -> customers.csv (data/)

The dashboard tabs that show computed metrics and charts (Overview,
Expense Analytics, Financial Health) talk directly to FinanceEngine and
work even without a Groq API key. Only the AI Insights and Chat tabs
require the LLM, since those are where natural-language reasoning and
tool-calling actually happen.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from agent import FinanceAgent
from config import APP_ICON, APP_TITLE, CURRENCY_SYMBOL, EMERGENCY_FUND_MONTHS, IDEAL_SAVINGS_PCT
from finance_engine import FinanceEngine
from llm import MissingAPIKeyError
from memory import MemoryStore
import visualizer as viz
from utils import build_csv_report, build_pdf_report, format_currency, format_pct, get_customer_list, load_customer_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css(dark_mode: bool) -> None:
    if dark_mode:
        surface, page, primary, secondary, border, card_grad = (
            "#1a1a19", "#0d0d0d", "#ffffff", "#c3c2b7", "rgba(255,255,255,0.10)",
            "linear-gradient(135deg, #1c2b3f 0%, #1a1a19 100%)",
        )
    else:
        surface, page, primary, secondary, border, card_grad = (
            "#fcfcfb", "#f9f9f7", "#0b0b0b", "#52514e", "rgba(11,11,11,0.10)",
            "linear-gradient(135deg, #eaf2fd 0%, #fcfcfb 100%)",
        )

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {page}; }}
        .kpi-card {{
            background: {card_grad};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 6px;
        }}
        .kpi-label {{
            font-size: 0.78rem; font-weight: 600; letter-spacing: .02em;
            text-transform: uppercase; color: {secondary}; margin-bottom: 6px;
        }}
        .kpi-value {{ font-size: 1.55rem; font-weight: 700; color: {primary}; }}
        .kpi-sub {{ font-size: 0.8rem; color: {secondary}; margin-top: 2px; }}
        .insight-card {{
            background: {surface}; border: 1px solid {border}; border-radius: 12px;
            padding: 14px 16px; margin-bottom: 12px;
        }}
        section[data-testid="stSidebar"] {{ border-right: 1px solid {border}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>{sub_html}</div>"""


# --------------------------------------------------------------------------- #
# Cached data / session state
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    return load_customer_data()


def _init_session_state() -> None:
    defaults = {
        "dark_mode": False,
        "customer_id": None,
        "selected_month": None,
        "memory_store": MemoryStore(),
        "agents": {},
        "last_insights": None,
        "last_report_summary": None,
        "savings_goal_amount": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_agent(engine: FinanceEngine) -> FinanceAgent | None:
    """Lazily build (and cache in session_state) a FinanceAgent for the
    currently selected customer. Returns None if GROQ_API_KEY is missing."""
    cid = engine.customer_id
    if cid in st.session_state.agents:
        return st.session_state.agents[cid]
    try:
        memory = st.session_state.memory_store.get(cid)
        agent = FinanceAgent(engine, memory)
        st.session_state.agents[cid] = agent
        return agent
    except MissingAPIKeyError:
        return None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(df: pd.DataFrame) -> tuple[str, str]:
    st.sidebar.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.sidebar.caption("AI-powered banking financial coach")
    st.sidebar.divider()

    customers = get_customer_list(df)
    customer_labels = {c["customer_id"]: f"{c['customer_name']} ({c['customer_id']})" for c in customers}
    default_idx = 0
    if st.session_state.customer_id in customer_labels:
        default_idx = list(customer_labels.keys()).index(st.session_state.customer_id)

    customer_id = st.sidebar.selectbox(
        "Select Customer",
        options=list(customer_labels.keys()),
        format_func=lambda cid: customer_labels[cid],
        index=default_idx,
    )

    if customer_id != st.session_state.customer_id:
        st.session_state.customer_id = customer_id
        st.session_state.selected_month = None  # reset month on customer switch

    engine_preview = FinanceEngine(df, customer_id)
    months = engine_preview.available_months
    month_idx = len(months) - 1
    if st.session_state.selected_month in months:
        month_idx = months.index(st.session_state.selected_month)

    month = st.sidebar.selectbox("Select Month", options=months, index=month_idx)
    st.session_state.selected_month = month

    st.sidebar.divider()
    st.session_state.dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

    if st.sidebar.button("🔄 Reset Chat & Memory", use_container_width=True):
        st.session_state.memory_store.reset(customer_id)
        st.session_state.agents.pop(customer_id, None)
        st.session_state.last_insights = None
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Built with Groq · Llama 3.3 · LangChain · Streamlit")

    return customer_id, month


# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
def render_kpi_row(engine: FinanceEngine, month: str) -> dict:
    summary = engine.monthly_summary(month)
    health = engine.financial_health_score(month)
    budget = engine.budget_analysis(month)

    cols = st.columns(4)
    with cols[0]:
        st.markdown(kpi_card("Monthly Income", format_currency(summary["monthly_income"])), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(kpi_card("Monthly Expense", format_currency(summary["total_expense"]),
                              f"{format_pct(summary['expense_ratio_pct'])} of income"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(kpi_card("Net Savings", format_currency(summary["net_savings"])), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(kpi_card("Savings %", format_pct(summary["savings_pct"]),
                              f"Target {format_pct(IDEAL_SAVINGS_PCT * 100)}"), unsafe_allow_html=True)

    cols2 = st.columns(4)
    with cols2[0]:
        st.markdown(kpi_card("Financial Health Score", f"{health['financial_health_score']} / 100", health["rating"]),
                    unsafe_allow_html=True)
    with cols2[1]:
        st.markdown(kpi_card("Expense Ratio", format_pct(summary["expense_ratio_pct"])), unsafe_allow_html=True)
    with cols2[2]:
        st.markdown(kpi_card("Highest Expense Category", summary["highest_expense_category"],
                              format_currency(summary["highest_expense_amount"])), unsafe_allow_html=True)
    with cols2[3]:
        st.markdown(kpi_card("Budget Utilization", format_pct(budget["budget_utilization_pct"]),
                              f"{budget['violation_count']} categories over budget"), unsafe_allow_html=True)

    return {"summary": summary, "health": health, "budget": budget}


# --------------------------------------------------------------------------- #
# Smart alerts (deterministic, no LLM)
# --------------------------------------------------------------------------- #
def render_smart_alerts(engine: FinanceEngine, month: str) -> None:
    budget = engine.budget_analysis(month)
    emergency = engine.emergency_fund_estimate(month)
    summary = engine.monthly_summary(month)

    if budget["violation_count"] > 0:
        top = budget["budget_violations"][0]
        st.warning(
            f"⚠️ Budget alert: **{top['category']}** is "
            f"{top['overspend_pct_points']:.1f} points over its recommended budget share "
            f"this month, along with {budget['violation_count'] - 1} other categor"
            f"{'y' if budget['violation_count'] - 1 == 1 else 'ies'}."
        )
    if not emergency["is_adequately_covered"]:
        st.warning(
            f"🛟 Emergency fund alert: current coverage is only "
            f"**{emergency['months_covered']} months** vs the recommended "
            f"{EMERGENCY_FUND_MONTHS} months (shortfall: {format_currency(emergency['shortfall'])})."
        )
    if summary["net_savings"] < 0:
        st.error("🚨 Negative cash flow this month — expenses exceeded income.")
    if budget["violation_count"] == 0 and emergency["is_adequately_covered"] and summary["net_savings"] >= 0:
        st.success("✅ No alerts — this month looks financially healthy.")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_overview(engine: FinanceEngine, month: str, dark_mode: bool) -> None:
    render_smart_alerts(engine, month)
    st.markdown("#### Income vs. Expense vs. Savings Trend")
    trend = engine.cash_flow_analysis()
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.monthly_expense_line(trend["months"], trend["outflow"], dark_mode), use_container_width=True)
    with col2:
        savings_history = engine.savings_growth()["monthly_savings_history"]
        months = [s["month"] for s in savings_history]
        savings = [s["savings"] for s in savings_history]
        st.plotly_chart(viz.savings_trend_line(months, savings, dark_mode), use_container_width=True)

    st.markdown("#### Quick Facts")
    profile = engine.customer_profile()
    growth = engine.savings_growth()
    c1, c2, c3 = st.columns(3)
    c1.metric("Customer", profile["name"])
    c1.caption(f"{profile['occupation']}, age {profile['age']}")
    c2.metric("Avg Monthly Savings (12mo)", format_currency(growth["average_monthly_savings"]))
    c3.metric("Total Savings Accumulated", format_currency(growth["total_savings_over_period"]))


def tab_expense_analytics(engine: FinanceEngine, month: str, dark_mode: bool) -> None:
    summary = engine.monthly_summary(month)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.expense_distribution_pie(summary["category_expenses"], dark_mode), use_container_width=True)
    with col2:
        st.plotly_chart(viz.expense_categories_bar(summary["category_expenses"], dark_mode), use_container_width=True)

    st.plotly_chart(viz.expense_treemap(summary["category_expenses"], dark_mode), use_container_width=True)

    full_history = engine.full_history_df()
    st.plotly_chart(viz.category_comparison_stacked_bar(full_history, dark_mode), use_container_width=True)
    st.plotly_chart(viz.monthly_spending_heatmap(full_history, dark_mode), use_container_width=True)


def tab_financial_health(engine: FinanceEngine, month: str, dark_mode: bool) -> None:
    health = engine.financial_health_score(month)
    summary = engine.monthly_summary(month)
    budget = engine.budget_analysis(month)
    emergency = engine.emergency_fund_estimate(month)
    cash_flow = engine.cash_flow_analysis()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.financial_health_gauge(health["financial_health_score"], dark_mode), use_container_width=True)
    with col2:
        st.plotly_chart(viz.savings_pct_gauge(summary["savings_pct"], IDEAL_SAVINGS_PCT * 100, dark_mode), use_container_width=True)

    st.markdown("#### Health Score Components")
    comp = health["component_scores"]
    ccols = st.columns(4)
    ccols[0].metric("Savings Rate", f"{comp['savings_rate_score']}/100")
    ccols[1].metric("Budget Adherence", f"{comp['budget_adherence_score']}/100")
    ccols[2].metric("Expense Stability", f"{comp['expense_stability_score']}/100")
    ccols[3].metric("Emergency Fund", f"{comp['emergency_fund_score']}/100")

    st.markdown("#### Budget Violations")
    if budget["budget_violations"]:
        st.dataframe(pd.DataFrame(budget["budget_violations"]), use_container_width=True, hide_index=True)
    else:
        st.success("No budget violations this month.")

    st.markdown("#### Emergency Fund Status")
    e1, e2, e3 = st.columns(3)
    e1.metric("Months Covered", f"{emergency['months_covered']}")
    e2.metric("Recommended Fund", format_currency(emergency["recommended_fund_amount"]))
    e3.metric("Shortfall", format_currency(emergency["shortfall"]))

    st.markdown("#### Savings Goal Tracker")
    goal = st.number_input(
        "Set a savings goal (₹)", min_value=0.0, step=5000.0,
        value=st.session_state.savings_goal_amount or 100000.0,
    )
    st.session_state.savings_goal_amount = goal
    growth = engine.savings_growth()
    accumulated = growth["total_savings_over_period"]
    avg_monthly = growth["average_monthly_savings"]
    progress = min(1.0, max(0.0, accumulated / goal)) if goal else 0.0
    st.progress(progress, text=f"{format_currency(accumulated)} / {format_currency(goal)} ({progress*100:.1f}%)")
    if avg_monthly > 0 and accumulated < goal:
        months_left = round((goal - accumulated) / avg_monthly, 1)
        st.caption(f"At your average monthly savings rate, you'll reach this goal in ~{months_left} months.")

    st.markdown("#### Cash Flow Analysis")
    st.caption(
        f"{cash_flow['positive_cash_flow_months']} positive months · "
        f"{cash_flow['negative_cash_flow_months']} negative months · "
        f"avg net flow {format_currency(cash_flow['average_net_cash_flow'])}"
    )


def tab_ai_insights(engine: FinanceEngine, month: str) -> None:
    agent = get_agent(engine)
    if agent is None:
        st.info("🔑 Set GROQ_API_KEY in your .env file to enable AI-generated insights.")
        return

    if st.button("✨ Generate AI Insights", type="primary"):
        with st.spinner("Analyzing financial data with Llama..."):
            try:
                text, trace = agent.generate_insights(month)
                st.session_state.last_insights = {"text": text, "trace": trace, "month": month}
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to generate insights: {exc}")

    result = st.session_state.last_insights
    if result:
        st.markdown(f'<div class="insight-card">{result["text"]}</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download AI Insights (.md)",
            data=result["text"].encode("utf-8"),
            file_name=f"ai_insights_{engine.customer_id}_{result['month']}.md",
            mime="text/markdown",
        )
        with st.expander("🔧 Tool calls used to generate this"):
            for call in result["trace"]:
                st.markdown(f"**{call['tool']}**({call['args']})")
                st.json(call["result"])
    else:
        st.caption("Click the button above to generate grounded, tool-based financial insights.")


def tab_chat(engine: FinanceEngine, month: str) -> None:
    agent = get_agent(engine)
    if agent is None:
        st.info("🔑 Set GROQ_API_KEY in your .env file to enable the chat assistant.")
        return

    memory = st.session_state.memory_store.get(engine.customer_id)

    st.caption(
        "Ask things like: *\"Where am I overspending?\"*, *\"Can I afford a "
        "₹50,000 vacation?\"*, *\"Compare this month with last month.\"*"
    )

    for msg in memory.get_history():
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if history_text := "\n\n".join(f"{m.type}: {m.content}" for m in memory.get_history()):
        st.download_button(
            "⬇️ Export Chat History", data=history_text.encode("utf-8"),
            file_name=f"chat_history_{engine.customer_id}.txt", mime="text/plain",
        )

    user_input = st.chat_input("Ask your finance question...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply, trace = agent.chat(user_input)
                except Exception as exc:  # noqa: BLE001
                    reply, trace = f"Sorry, something went wrong: {exc}", []
                st.markdown(reply)
                if trace:
                    with st.expander("🔧 Tools used"):
                        for call in trace:
                            st.markdown(f"**{call['tool']}**({call['args']})")
        st.rerun()


def tab_reports(engine: FinanceEngine, month: str) -> None:
    summary = engine.monthly_summary(month)
    health = engine.financial_health_score(month)
    profile = engine.customer_profile()
    agent = get_agent(engine)

    st.markdown("#### Generate Report")
    ai_summary = ""
    recommendations: list[str] = []

    if agent is not None and st.button("🧠 Generate AI Summary for Report"):
        with st.spinner("Summarizing with Llama..."):
            try:
                ai_summary, _ = agent.generate_report_summary(month)
                st.session_state.last_report_summary = ai_summary
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to generate summary: {exc}")

    ai_summary = st.session_state.get("last_report_summary") or ai_summary
    if ai_summary:
        st.markdown(f'<div class="insight-card">{ai_summary}</div>', unsafe_allow_html=True)

    budget = engine.budget_analysis(month)
    for v in budget["budget_violations"][:3]:
        recommendations.append(
            f"Reduce {v['category']} spending — currently {v['spent_pct']}% of income "
            f"vs recommended {v['recommended_pct']}%."
        )
    if not recommendations:
        recommendations.append("Spending is within recommended budget shares across all categories — keep it up.")

    report_payload = engine.recommendation_inputs(month)

    col1, col2 = st.columns(2)
    with col1:
        csv_bytes = build_csv_report(report_payload)
        st.download_button(
            "⬇️ Download CSV Report", data=csv_bytes,
            file_name=f"finance_report_{engine.customer_id}_{month}.csv", mime="text/csv",
            use_container_width=True,
        )
    with col2:
        pdf_bytes = build_pdf_report(
            customer_name=profile["name"], month=month, summary=summary, health=health,
            recommendations=recommendations, ai_summary=ai_summary,
        )
        st.download_button(
            "⬇️ Download PDF Report", data=pdf_bytes,
            file_name=f"finance_report_{engine.customer_id}_{month}.pdf", mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("#### Report Preview")
    st.json(report_payload)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _init_session_state()

    try:
        df = _load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    customer_id, month = render_sidebar(df)
    inject_css(st.session_state.dark_mode)
    engine = FinanceEngine(df, customer_id)

    st.title(f"{APP_ICON} {APP_TITLE}")
    profile = engine.customer_profile()
    st.caption(f"Financial dashboard for **{profile['name']}** — {profile['occupation']} · {month}")

    render_kpi_row(engine, month)
    st.divider()

    tabs = st.tabs([
        "📊 Overview", "🧾 Expense Analytics", "❤️ Financial Health",
        "🤖 AI Insights", "💬 Chat with Finance AI", "📄 Reports",
    ])
    dark_mode = st.session_state.dark_mode

    with tabs[0]:
        tab_overview(engine, month, dark_mode)
    with tabs[1]:
        tab_expense_analytics(engine, month, dark_mode)
    with tabs[2]:
        tab_financial_health(engine, month, dark_mode)
    with tabs[3]:
        tab_ai_insights(engine, month)
    with tabs[4]:
        tab_chat(engine, month)
    with tabs[5]:
        tab_reports(engine, month)


if __name__ == "__main__":
    main()
