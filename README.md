# 💰 Personal Finance AI Advisor

An **agentic AI application** that helps banking customers understand their
monthly financial health — built with **Groq (Llama 3.3 70B)**, **LangChain
tool calling**, and **Streamlit**.

This is not a chatbot bolted onto a dashboard. It is a small agentic system:
the LLM reasons, decides which financial tools to call, and explains the
results — but it **never does arithmetic itself**. Every number on screen is
produced by a deterministic Python engine; the LLM's only job is to call the
right tool and explain what the numbers mean.

---

## Overview

| | |
|---|---|
| **Reasoning engine** | Groq-hosted Llama 3.3 70B, via `langchain-groq` |
| **Calculations** | 100% deterministic Python (`finance_engine.py`) — the LLM never computes |
| **Tool calling** | 13 LangChain tools bound per-customer, invoked by the model at runtime |
| **Memory** | Conversation buffer + structured context extraction (stated income/goals) |
| **UI** | Streamlit, 6 tabs, light/dark theme, Plotly charts |
| **Data** | 10 synthetic customers × 12 months of realistic transaction history |

---

## Architecture

```
                    ┌─────────────────────┐
                    │     Streamlit UI     │  app.py
                    │  (6 dashboard tabs)  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    Finance AI Agent   │  agent.py
                    │  (tool-calling loop)  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Groq · Llama 3.3    │  llm.py
                    │  (decides which tool  │
                    │   to call, and how    │
                    │   to explain results) │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    Agent Tool Layer    │  tools.py
                    │ (13 StructuredTools)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Deterministic Finance │  finance_engine.py
                    │        Engine          │  (all math lives here)
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   customers.csv        │  data/
                    │ (10 customers × 12mo)  │
                    └─────────────────────────┘
```

The LLM sees only the structured JSON a tool returns — never raw transaction
rows — so every claim it makes is traceable back to a specific calculation.

---

## Features

### Deterministic Financial Engine (`finance_engine.py`)
Total income/expense, net savings, savings %, expense ratio, category
breakdowns, highest/lowest category, monthly trend, savings growth, budget
utilization, emergency fund estimate, cash flow analysis, month-over-month
comparison, affordability checks, and a weighted **Financial Health Score
(0–100)**.

### Agentic Tool Calling (`tools.py` + `agent.py`)
13 tools the LLM can invoke on demand: expense analyzer, savings calculator,
savings growth, budget analyzer, trend analyzer, financial health score,
emergency fund estimate, cash flow analysis, recommendation-input bundler,
monthly summary, affordability check, month comparison, and customer profile.
The agent runs a standard bind-tools reasoning loop (tool calls → tool
results → final answer), capped at 6 iterations.

### Conversational Memory (`memory.py`)
A buffer of the full chat transcript, replayed to the model every turn, plus
lightweight structured extraction of stated income/savings goals mentioned in
free text (e.g. *"my income is ₹90,000"* → later *"can I save ₹25,000 a
month?"* is understood in context). Memory is scoped per customer so
switching customers never leaks context.

### Dashboard (`app.py`)
- **Sidebar:** customer selector, month selector, dark-mode toggle, reset chat.
- **KPI row:** income, expense, savings, savings %, health score, expense
  ratio, highest expense category, budget utilization.
- **📊 Overview** — income/expense/savings trend, smart alerts, quick facts.
- **🧾 Expense Analytics** — pie, bar, treemap, stacked bar, heatmap.
- **❤️ Financial Health** — health score & savings gauges, component
  breakdown, budget violations, emergency fund status, **savings goal
  tracker**, cash flow summary.
- **🤖 AI Insights** — LLM-generated, tool-grounded insights (Observation /
  Reason / Recommendation / Expected Benefit), downloadable.
- **💬 Chat with Finance AI** — free-form Q&A with full tool access and
  memory; exportable chat history.
- **📄 Reports** — AI executive summary, downloadable CSV and PDF reports.

### Deterministic tabs work without an API key
Overview, Expense Analytics, and Financial Health need no LLM at all — only
AI Insights and Chat require `GROQ_API_KEY`. If it's missing, those two tabs
show a friendly setup prompt instead of crashing.

---

## Project Structure

```
personal_finance_ai_agent/
│
├── app.py              # Streamlit UI entry point (6 tabs)
├── config.py            # Central config: paths, model settings, constants
├── agent.py              # Tool-calling agent orchestration loop
├── llm.py                 # Groq (ChatGroq) client factory
├── prompts.py              # System prompt + structured task prompts
├── tools.py                 # LangChain tool wrappers around the finance engine
├── finance_engine.py         # All deterministic financial calculations
├── memory.py                  # Conversation buffer + structured context memory
├── visualizer.py                # Plotly chart builders (light/dark themed)
├── sample_data.py                 # Synthetic customer/transaction generator
├── utils.py                        # Data loading, formatting, CSV/PDF reports
├── requirements.txt
├── README.md
│
├── data/
│   └── customers.csv       # Generated by sample_data.py
├── reports/                 # Downloaded CSV/PDF reports land here
├── assets/
└── screenshots/
```

---

## Installation

### 1. Clone / open the project
```bash
cd personal_finance_ai_agent
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the sample dataset
```bash
python sample_data.py
```
This writes `data/customers.csv` — 10 customers × 12 months of realistic,
seasonally-varying transaction history.

---

## Groq API Setup

1. Create a free account at [console.groq.com](https://console.groq.com).
2. Generate an API key.
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Fill in your key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes (for AI tabs) | — | Your Groq API key |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq-hosted Llama model to use |
| `LLM_TEMPERATURE` | No | `0.3` | Sampling temperature |
| `LLM_MAX_TOKENS` | No | `1500` | Max tokens per response |

---

## Running the Application

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`). Select a
customer and month from the sidebar, and explore the tabs.

---

## Screenshots

> Place screenshots in `screenshots/` and reference them here, e.g.:
>
> `![Overview tab](screenshots/overview.png)`
> `![AI Insights](screenshots/insights.png)`
> `![Chat](screenshots/chat.png)`

---

## Future Enhancements

- Persist chat memory and goals to a database instead of in-memory session state
- Multi-turn goal planning (e.g. automatic savings plans toward a target date)
- Bank account aggregation via Plaid/Account Aggregator instead of synthetic CSV
- Streaming token-by-token responses in the chat tab
- Role-based multi-customer view for relationship managers
- Anomaly detection tool for unusual single transactions (not just category drift)
- Voice input/output for the chat assistant

---

## Troubleshooting

**"GROQ_API_KEY is not set" message in AI tabs**
Create `.env` from `.env.example` and add a valid key, then restart Streamlit.

**`FileNotFoundError: Customer dataset not found`**
Run `python sample_data.py` before `streamlit run app.py`.

**PDF report fails to download**
Ensure `fpdf2` (not the abandoned `fpdf`) is installed — check with
`pip show fpdf2`. The report generator sanitizes `₹` to `Rs.` internally
since PDF core fonts don't support the Rupee glyph.

**Charts look unstyled / wrong colors in dark mode**
Toggle dark mode from the sidebar rather than your OS theme — charts are
explicitly re-themed by the toggle, not by `prefers-color-scheme`.

**Groq rate limit / model errors**
Check your plan's rate limits at [console.groq.com](https://console.groq.com).
Lower `LLM_MAX_TOKENS` in `.env` if you're hitting token-per-minute limits.

---

## License

MIT License — free to use, modify, and extend for personal or commercial projects.
