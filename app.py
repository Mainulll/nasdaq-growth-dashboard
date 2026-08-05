from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from agents import analyse_news, news_critic, portfolio_critic, score_holding
from data_sources import (
    current_audusd,
    download_prices,
    finnhub_company_news,
    finnhub_quotes,
    fundamentals_table,
    gdelt_news,
)
from portfolio_engine import (
    BENCHMARK_WEIGHTS,
    allocate_monthly,
    benchmark_index,
    current_weights,
    market_regime,
    momentum,
    sell_flags,
    unrealised_pnl,
)

ROOT = Path(__file__).resolve().parent
PORTFOLIO_FILE = ROOT / "portfolio.csv"
SNAPSHOT_FILE = ROOT / "data" / "latest_snapshot.json"

st.set_page_config(page_title="High-Risk Growth Portfolio", page_icon="📈", layout="wide")

st.title("High-Risk Growth Portfolio Research Dashboard")
st.caption("5–10 year horizon • AUD 2,000 monthly • long-only • balanced growth • maximum tolerated drawdown approximately 60%")

st.warning(
    "Research and decision support only—not personal financial advice or an execution system. "
    "Free market feeds can be delayed, incomplete or revised. Verify prices and orders in CommSec before trading."
)


def get_secret(name: str, default: str = "") -> str:
    """Read a secret without requiring a secrets file to exist.

    `st.secrets.get()` raises StreamlitSecretNotFoundError when no secrets.toml
    is present, which crashed the whole app on a fresh deploy before any content
    rendered. Secrets are optional here, so a missing file must degrade quietly.
    """
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: tuple[str, ...], period: str = "2y"):
    return download_prices(tickers, period=period)

@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(tickers: tuple[str, ...]):
    return fundamentals_table(tickers)

@st.cache_data(ttl=120, show_spinner=False)
def load_quotes(tickers: tuple[str, ...], finnhub_key: str | None):
    """Cache live quotes.

    Streamlit reruns the whole script on every widget interaction, so an
    uncached call here fired one Finnhub request per holding on each keystroke
    in the holdings editor — enough to exhaust the 60 requests/minute free tier
    within seconds. Two minutes is fresh enough for a monthly-contribution tool.
    """
    return finnhub_quotes(tickers, finnhub_key)


@st.cache_data(ttl=900, show_spinner=False)
def load_audusd():
    return current_audusd()


@st.cache_data(ttl=900, show_spinner=False)
def load_news(symbol: str, company: str, finnhub_key: str | None):
    frames = []
    f = finnhub_company_news(symbol, finnhub_key, days=7)
    if not f.empty:
        frames.append(f)
    g = gdelt_news(f'("{company}" OR "{symbol}") sourcecountry:US', days=7, max_records=35)
    if not g.empty:
        g["ticker"] = symbol
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    # Finnhub returns ISO timestamps and GDELT returns 20260805T120000Z, so a
    # plain string sort interleaved the two feeds into a meaningless order.
    merged["published"] = pd.to_datetime(merged["published"], format="mixed", utc=True, errors="coerce")
    return analyse_news(merged)

portfolio = pd.read_csv(PORTFOLIO_FILE)
portfolio["units"] = pd.to_numeric(portfolio["units"], errors="coerce").fillna(0.0)
portfolio["avg_cost_aud"] = pd.to_numeric(portfolio["avg_cost_aud"], errors="coerce").fillna(0.0)

with st.sidebar:
    st.header("Inputs")
    monthly_aud = st.number_input("Monthly contribution (AUD)", min_value=100.0, max_value=100000.0, value=2000.0, step=100.0)
    max_orders = st.slider("Maximum CommSec orders per month", 1, 6, 3)
    min_order_aud = st.number_input(
        "Minimum order size (AUD)", min_value=0.0, max_value=5000.0, value=250.0, step=50.0,
        help="Orders below this are merged away so brokerage and the 0.55% FX spread do not dominate the trade.",
    )
    finnhub_key = get_secret("FINNHUB_API_KEY")
    st.caption("Add FINNHUB_API_KEY in Streamlit secrets for live portfolio/watchlist quotes and company news.")
    st.subheader("Current holdings")
    edited = st.data_editor(
        portfolio[["ticker", "units", "avg_cost_aud", "target_weight", "max_weight"]],
        hide_index=True,
        disabled=["ticker", "target_weight", "max_weight"],
        width="stretch",
        num_rows="fixed",
    )
    portfolio["units"] = pd.to_numeric(edited["units"], errors="coerce").fillna(0.0)
    portfolio["avg_cost_aud"] = pd.to_numeric(edited["avg_cost_aud"], errors="coerce").fillna(0.0)
    st.caption("For the CASH row, enter your uninvested AUD balance in the units column.")
    st.download_button("Download updated holdings CSV", portfolio.to_csv(index=False), "portfolio_updated.csv", "text/csv")

market_tickers = tuple(t for t in portfolio["ticker"] if t != "CASH")
all_price_tickers = tuple(sorted(set(market_tickers) | set(BENCHMARK_WEIGHTS)))
prices, price_status = load_prices(all_price_tickers)
fundamentals = load_fundamentals(market_tickers)

live, live_status = load_quotes(market_tickers, finnhub_key)
delayed_quotes = prices.ffill().iloc[-1] if not prices.empty else pd.Series(dtype=float)
if not live.empty and "price" in live.columns:
    # Backfill any symbol Finnhub could not price, rather than dropping it to a
    # zero weight and silently understating the position.
    quotes = live["price"].dropna().combine_first(delayed_quotes)
else:
    quotes = delayed_quotes

audusd, fx_status = load_audusd()
if fx_status.source == "Fallback constant":
    st.error(fx_status.message)
regime = market_regime(prices)
current_w = current_weights(portfolio, quotes, audusd)

mom6 = momentum(prices, 126)
mom12 = momentum(prices, 252)

agent_rows = []
critic_score_map: dict[str, float] = {}
for _, p_row in portfolio[portfolio["ticker"] != "CASH"].iterrows():
    ticker = p_row["ticker"]
    f_row = fundamentals.loc[ticker] if ticker in fundamentals.index else pd.Series(dtype=object)
    decision = score_holding(f_row, mom6.get(ticker), mom12.get(ticker), str(p_row["risk_class"]))
    sector = str(f_row.get("sector") or p_row["sleeve"])
    sector_mask = portfolio["sleeve"].eq(p_row["sleeve"])
    sector_weight = float(current_w[sector_mask].sum()) if len(current_w) else 0.0
    critic = portfolio_critic(
        f_row,
        decision,
        float(current_w.loc[p_row.name]) if p_row.name in current_w.index else 0.0,
        float(p_row["target_weight"]),
        float(p_row["max_weight"]),
        sector_weight,
        str(p_row["risk_class"]),
    )
    critic_score_map[ticker] = critic.score
    agent_rows.append(
        {
            "ticker": ticker,
            "name": p_row["name"],
            "sleeve": p_row["sleeve"],
            "fundamental_score": decision.score,
            "fundamental_label": decision.label,
            "critic_score": critic.score,
            "critic_action": critic.label,
            "reasons": "; ".join(critic.reasons),
            "warnings": "; ".join(critic.warnings),
        }
    )
agent_table = pd.DataFrame(agent_rows).set_index("ticker")
critic_scores = pd.Series(critic_score_map)

allocation = allocate_monthly(
    portfolio, critic_scores, current_w, monthly_aud, regime,
    max_orders=max_orders, min_order_aud=min_order_aud,
)
sells = sell_flags(portfolio, critic_scores, current_w)
pnl = unrealised_pnl(portfolio, quotes, audusd)

# Header metrics
portfolio_value = float(pnl["market_value_aud"].sum()) if not pnl.empty else 0.0
total_pnl = float(pnl["pnl_aud"].sum()) if not pnl.empty else 0.0
cost_total = float(pnl["cost_basis_aud"].sum()) if not pnl.empty else 0.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Regime", regime.name)
m2.metric("Portfolio value", f"AUD {portfolio_value:,.0f}")
m3.metric(
    "Unrealised P&L",
    f"AUD {total_pnl:,.0f}",
    f"{(total_pnl / cost_total * 100):.1f}%" if cost_total > 0 else None,
)
m4.metric("AUD/USD", f"{audusd:.4f}")
m5.metric("Data mode", "Finnhub live" if not live.empty else "Delayed fallback")

st.caption(
    f"This month: invest AUD {monthly_aud * regime.contribution_invested:,.0f} · "
    f"hold AUD {monthly_aud * (1 - regime.contribution_invested):,.0f} as cash "
    f"({regime.name} regime, {regime.cash_target:.0%} cash target)."
)

st.caption("Regime evidence: " + " • ".join(regime.reasons))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Portfolio", "Fundamentals", "Monthly actions", "News critic", "Method and controls"])

with tab1:
    st.subheader("Target construction")
    target_plot = portfolio.copy()
    target_plot["target_pct"] = target_plot["target_weight"] * 100
    fig = px.sunburst(target_plot, path=["risk_class", "sleeve", "ticker"], values="target_pct", title="Target allocation by risk sleeve")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Holdings and unrealised P&L")
    if pnl.empty:
        st.info("Enter your units and average AUD cost in the sidebar to see position-level profit and loss.")
    else:
        st.dataframe(
            pnl[["ticker", "name", "units", "price_usd", "avg_cost_aud", "cost_basis_aud", "market_value_aud", "pnl_aud", "pnl_pct"]],
            hide_index=True,
            width="stretch",
            column_config={
                "price_usd": st.column_config.NumberColumn("Price (USD)", format="%.2f"),
                "avg_cost_aud": st.column_config.NumberColumn("Avg cost (AUD)", format="%.2f"),
                "cost_basis_aud": st.column_config.NumberColumn("Cost basis (AUD)", format="%.0f"),
                "market_value_aud": st.column_config.NumberColumn("Value (AUD)", format="%.0f"),
                "pnl_aud": st.column_config.NumberColumn("P&L (AUD)", format="%.0f"),
                "pnl_pct": st.column_config.NumberColumn("P&L %", format="%.1f%%"),
            },
        )
        st.caption("Cost basis uses the average AUD cost you entered. Market value converts the live USD price at the current AUD/USD rate, so it includes currency movement.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current estimated weights")
        view = portfolio[["ticker", "name", "target_weight", "max_weight"]].copy()
        view["current_weight"] = current_w.values
        view["price_usd"] = view["ticker"].map(quotes)
        view["target_weight"] = (view["target_weight"] * 100).round(2)
        view["max_weight"] = (view["max_weight"] * 100).round(2)
        view["current_weight"] = (view["current_weight"] * 100).round(2)
        st.dataframe(view, hide_index=True, width="stretch")
    with col2:
        st.subheader("Balanced benchmark")
        bench = benchmark_index(prices)
        if not bench.empty:
            st.line_chart(bench)
        st.write("50% QQQ / 30% SPY / 20% IWO")
        st.caption("The benchmark separates stock-selection skill from simply owning large-cap technology and small-cap growth beta.")

with tab2:
    st.subheader("Independent fundamental agent and critic")
    # Both frames carry a `name` column (portfolio.csv and the Yahoo record), so
    # an unsuffixed join raises and takes the whole tab down.
    display = agent_table.join(fundamentals, how="left", rsuffix="_data")
    preferred = [
        "name", "sleeve", "fundamental_score", "critic_score", "critic_action", "revenue_growth",
        "earnings_growth", "gross_margin", "operating_margin", "forward_pe", "price_to_sales",
        "free_cashflow", "total_debt", "beta", "warnings"
    ]
    display = display[[c for c in preferred if c in display.columns]].sort_values("critic_score", ascending=False)
    st.dataframe(display, width="stretch")
    st.caption("The critic can veto a superficially attractive growth score for concentration, leverage, negative cash flow, extreme valuation or speculative execution risk.")

with tab3:
    st.subheader("This month’s contribution plan")
    st.dataframe(allocation, hide_index=True, width="stretch")
    st.caption("The engine limits the number of trades because CommSec’s minimum US brokerage and FX conversion can materially erode small orders.")

    st.subheader("Sell / trim review")
    st.dataframe(sells, hide_index=True, width="stretch")
    st.info("A falling price is not a sell trigger by itself. Sales require a thesis break, persistent deterioration, excessive concentration or a materially better risk-adjusted alternative.")

with tab4:
    selected = st.selectbox("Company news set", market_tickers, index=0)
    company_name = portfolio.loc[portfolio["ticker"] == selected, "name"].iloc[0]
    news = load_news(selected, company_name, finnhub_key)
    critique = news_critic(news)
    n1, n2 = st.columns([1, 3])
    with n1:
        st.metric("News confidence", f"{critique.score:.0f}/100")
        st.write(critique.label)
        for w in critique.warnings:
            st.warning(w)
    with n2:
        if news.empty:
            st.info("No current news returned. Add a Finnhub key or try again later.")
        else:
            cols = [c for c in ["published", "ticker", "headline", "source", "sentiment", "url"] if c in news.columns]
            st.dataframe(news[cols].sort_values("published", ascending=False), hide_index=True, width="stretch", column_config={"url": st.column_config.LinkColumn("Source")})
    st.caption("News sentiment never creates a buy signal alone. It only changes confidence after source diversity, duplication, sensationalism and fundamental relevance checks.")

with tab5:
    st.markdown(
        """
### Decision hierarchy
1. **Survival:** no leverage, no options, no single speculative holding above 4–5%, and dynamic cash in weak regimes.
2. **Fundamentals:** revenue growth, margin quality, free cash flow, balance-sheet resilience and valuation.
3. **Portfolio fit:** target-weight gap, sector concentration, correlation and CommSec transaction costs.
4. **News:** only thesis-relevant events from diverse sources; sensational or duplicated coverage is penalised.
5. **Execution:** normally two or three monthly orders; use limits and verify the quote in CommSec.

### Hard sell conditions
- The core thesis is invalidated by regulation, technology displacement, accounting concerns or loss of competitive advantage.
- Revenue and cash-flow deterioration persists for at least two reporting periods without a credible explanation.
- A position breaches its maximum risk weight and the critic score is below 60.
- A superior alternative has a materially higher critic score and improves diversification after transaction costs and tax.

### Data design
- **Live/watchlist quotes and company news:** Finnhub when configured.
- **Historical prices and convenience fundamentals:** yfinance, treated as delayed research data.
- **Authoritative fundamentals:** validate against SEC EDGAR filings before material trades.
- **World-event narrative:** GDELT, with an independent source-quality critic.
- **Macro series:** FRED key can be added for rates, inflation, credit conditions and labour data.
"""
    )

    st.subheader("Last scheduled snapshot")
    if SNAPSHOT_FILE.exists():
        try:
            snapshot = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
            st.caption(f"Generated {snapshot.get('generated_at', 'unknown')} by the GitHub Actions job.")
            snap_regime = snapshot.get("regime", {})
            st.write(
                f"Regime at snapshot: **{snap_regime.get('name', 'unknown')}** "
                f"(cash target {snap_regime.get('cash_target', 0):.0%})"
            )
            st.dataframe(pd.DataFrame(snapshot.get("monthly_allocation", [])), hide_index=True, width="stretch")
            st.caption("This is the unattended overnight run. It should broadly agree with the live figures above; a large divergence means one of the feeds is failing.")
        except Exception as exc:
            st.info(f"Snapshot file could not be read: {exc}")
    else:
        st.info("No snapshot committed yet. The scheduled GitHub Actions job writes data/latest_snapshot.json.")

st.divider()
st.caption(
    f"Historical source: {price_status.source}. {price_status.message}. "
    f"Live source: {live_status.message}. FX: {fx_status.message}."
)
