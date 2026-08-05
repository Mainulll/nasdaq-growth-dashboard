from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agents import portfolio_critic, score_holding
from data_sources import current_audusd, download_prices, finnhub_quotes, fundamentals_table
from portfolio_engine import BENCHMARK_WEIGHTS, allocate_monthly, current_weights, market_regime, momentum, sell_flags

ROOT = Path(__file__).resolve().parent
portfolio = pd.read_csv(ROOT / "portfolio.csv")
portfolio["units"] = pd.to_numeric(portfolio["units"], errors="coerce").fillna(0.0)
portfolio["avg_cost_aud"] = pd.to_numeric(portfolio["avg_cost_aud"], errors="coerce").fillna(0.0)
market_tickers = [t for t in portfolio["ticker"] if t != "CASH"]
price_tickers = sorted(set(market_tickers) | set(BENCHMARK_WEIGHTS))
prices, price_status = download_prices(price_tickers, period="2y")
fundamentals = fundamentals_table(market_tickers)
quotes_live, live_status = finnhub_quotes(market_tickers, os.getenv("FINNHUB_API_KEY"))
delayed = prices.ffill().iloc[-1] if not prices.empty else pd.Series(dtype=float)
quotes = (
    quotes_live["price"].dropna().combine_first(delayed)
    if not quotes_live.empty and "price" in quotes_live
    else delayed
)
audusd, fx_status = current_audusd()
regime = market_regime(prices)
current_w = current_weights(portfolio, quotes, audusd)
mom6, mom12 = momentum(prices, 126), momentum(prices, 252)

scores = {}
records = []
for _, p in portfolio[portfolio["ticker"] != "CASH"].iterrows():
    t = p["ticker"]
    f = fundamentals.loc[t] if t in fundamentals.index else pd.Series(dtype=object)
    base = score_holding(f, mom6.get(t), mom12.get(t), str(p["risk_class"]))
    # Mirror the dashboard: score against the real current and sleeve weights so
    # the unattended run cannot disagree with what the app shows.
    sleeve_weight = float(current_w[portfolio["sleeve"].eq(p["sleeve"])].sum()) if len(current_w) else 0.0
    critic = portfolio_critic(
        f,
        base,
        float(current_w.loc[p.name]) if p.name in current_w.index else 0.0,
        float(p["target_weight"]),
        float(p["max_weight"]),
        sleeve_weight,
        str(p["risk_class"]),
    )
    scores[t] = critic.score
    records.append({"ticker": t, "score": critic.score, "action": critic.label, "warnings": critic.warnings})

score_s = pd.Series(scores)
allocation = allocate_monthly(portfolio, score_s, current_w, 2000.0, regime, max_orders=3)
sells = sell_flags(portfolio, score_s, current_w)

def jsonable(value):
    """Convert NaN/Inf to null so the committed file is valid JSON.

    json.dump writes a bare NaN token, which the JSON spec does not allow and
    strict parsers (including JSON.parse) reject outright. The snapshot is a
    published artefact, so it has to parse everywhere.
    """
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if pd.isna(value):
        return None
    return str(value)


snapshot = jsonable(
    {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regime": {"name": regime.name, "cash_target": regime.cash_target, "invested_fraction": regime.contribution_invested, "reasons": regime.reasons},
        "audusd": audusd,
        "fx_status": fx_status.message,
        "data_status": price_status.message,
        "live_status": live_status.message,
        "scores": records,
        "monthly_allocation": allocation.to_dict(orient="records"),
        "sell_review": sells.to_dict(orient="records"),
    }
)
(ROOT / "data").mkdir(exist_ok=True)
with open(ROOT / "data" / "latest_snapshot.json", "w", encoding="utf-8") as f:
    json.dump(snapshot, f, indent=2, allow_nan=False)
print(json.dumps(snapshot, indent=2, allow_nan=False))
