from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BENCHMARK_WEIGHTS = {"QQQ": 0.50, "SPY": 0.30, "IWO": 0.20}


@dataclass
class Regime:
    name: str
    cash_target: float
    contribution_invested: float
    reasons: list[str]


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="all") if not prices.empty else pd.DataFrame()


def momentum(prices: pd.DataFrame, periods: int) -> pd.Series:
    if prices.empty or len(prices) <= periods:
        return pd.Series(dtype=float)
    return prices.iloc[-1] / prices.iloc[-periods - 1] - 1


def market_regime(prices: pd.DataFrame) -> Regime:
    reasons: list[str] = []
    if prices.empty or "QQQ" not in prices:
        return Regime("Neutral / insufficient data", 0.10, 0.75, ["Benchmark data unavailable"])
    qqq = prices["QQQ"].dropna()
    if len(qqq) < 200:
        return Regime("Neutral / limited history", 0.10, 0.75, ["Less than 200 sessions of QQQ data"])
    last = float(qqq.iloc[-1])
    ma50 = float(qqq.rolling(50).mean().iloc[-1])
    ma200 = float(qqq.rolling(200).mean().iloc[-1])
    high = float(qqq.cummax().iloc[-1])
    drawdown = last / high - 1
    vol = float(qqq.pct_change(fill_method=None).rolling(20).std().iloc[-1] * np.sqrt(252))

    score = 0
    if last > ma200:
        score += 1
        reasons.append("QQQ is above its 200-day average")
    else:
        score -= 1
        reasons.append("QQQ is below its 200-day average")
    if ma50 > ma200:
        score += 1
        reasons.append("50-day trend is above 200-day trend")
    else:
        score -= 1
        reasons.append("50-day trend is below 200-day trend")
    if drawdown < -0.20:
        score -= 1
        reasons.append(f"QQQ drawdown is {drawdown:.1%}")
    if vol > 0.35:
        score -= 1
        reasons.append(f"Annualised 20-day volatility is {vol:.1%}")

    if score >= 2:
        return Regime("Risk-on", 0.04, 0.95, reasons)
    if score == 1:
        return Regime("Constructive", 0.08, 0.85, reasons)
    if score == 0:
        return Regime("Neutral", 0.15, 0.70, reasons)
    if score == -1:
        return Regime("Risk-off", 0.25, 0.45, reasons)
    return Regime("Defensive", 0.40, 0.25, reasons)


def benchmark_index(prices: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    weights = weights or BENCHMARK_WEIGHTS
    available = [t for t in weights if t in prices.columns]
    if not available:
        return pd.Series(dtype=float)
    normalised_weights = np.array([weights[t] for t in available], dtype=float)
    normalised_weights /= normalised_weights.sum()
    norm = prices[available].dropna(how="all").ffill().dropna()
    norm = norm / norm.iloc[0]
    return pd.Series(norm.values @ normalised_weights, index=norm.index, name="Balanced benchmark")


def position_values_aud(portfolio: pd.DataFrame, quotes: pd.Series, audusd: float) -> pd.Series:
    """Market value of every row in AUD.

    The CASH row carries its balance directly in `units`, denominated in AUD.
    Valuing it at zero previously excluded it from the denominator, so the cash
    weight always read 0% and the regime cash target could never be satisfied.
    """
    values = []
    for _, row in portfolio.iterrows():
        ticker = row["ticker"]
        units = float(row.get("units", 0) or 0)
        if ticker == "CASH":
            values.append(max(units, 0.0))
            continue
        px = float(quotes.get(ticker, np.nan))
        values.append(units * px / audusd if np.isfinite(px) and audusd > 0 else 0.0)
    return pd.Series(values, index=portfolio.index, dtype=float)


def current_weights(portfolio: pd.DataFrame, quotes: pd.Series, audusd: float) -> pd.Series:
    value_s = position_values_aud(portfolio, quotes, audusd)
    total = float(value_s.sum())
    return value_s / total if total > 0 else pd.Series(0.0, index=portfolio.index)


def unrealised_pnl(portfolio: pd.DataFrame, quotes: pd.Series, audusd: float) -> pd.DataFrame:
    """Cost-versus-market for each holding.

    `avg_cost_aud` was collected in the sidebar but never used anywhere, so the
    dashboard could not answer the most basic question a holder has: am I up or
    down, and by how much?
    """
    values = position_values_aud(portfolio, quotes, audusd)
    out = portfolio[["ticker", "name"]].copy()
    out["units"] = pd.to_numeric(portfolio["units"], errors="coerce").fillna(0.0)
    out["price_usd"] = out["ticker"].map(quotes)
    out["market_value_aud"] = values
    out["avg_cost_aud"] = pd.to_numeric(portfolio["avg_cost_aud"], errors="coerce").fillna(0.0)
    out["cost_basis_aud"] = out["units"] * out["avg_cost_aud"]
    out.loc[out["ticker"] == "CASH", "cost_basis_aud"] = out.loc[out["ticker"] == "CASH", "market_value_aud"]
    out["pnl_aud"] = out["market_value_aud"] - out["cost_basis_aud"]
    out["pnl_pct"] = np.where(
        out["cost_basis_aud"] > 0,
        out["pnl_aud"] / out["cost_basis_aud"].replace(0, np.nan) * 100,
        np.nan,
    )
    return out[out["units"] > 0].reset_index(drop=True)


def allocate_monthly(
    portfolio: pd.DataFrame,
    agent_scores: pd.Series,
    current_weight: pd.Series,
    monthly_aud: float,
    regime: Regime,
    max_orders: int = 3,
    min_order_aud: float = 250.0,
) -> pd.DataFrame:
    p = portfolio.copy()
    p["current_weight"] = current_weight.reindex(p.index).fillna(0.0)
    p["agent_score"] = agent_scores.reindex(p["ticker"]).values if len(agent_scores) else 50.0
    p["gap"] = (p["target_weight"] - p["current_weight"]).clip(lower=0)
    p["quality_multiplier"] = ((p["agent_score"] - 35) / 65).clip(lower=0.05, upper=1.0)
    p["priority"] = p["gap"] * p["quality_multiplier"]
    p.loc[p["ticker"] == "CASH", "priority"] = 0
    p.loc[p["agent_score"] < 45, "priority"] *= 0.15
    p.loc[p["risk_class"] == "speculative", "priority"] *= 0.70
    candidates = p[p["priority"] > 0].nlargest(max_orders, "priority").copy()

    investable = monthly_aud * regime.contribution_invested

    if candidates.empty:
        return pd.DataFrame([{"ticker": "CASH", "allocation_aud": monthly_aud, "reason": "No qualifying additions"}])

    def split(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["allocation_aud"] = frame["priority"] / frame["priority"].sum() * investable
        speculative_cap = monthly_aud * 0.25
        spec = frame["risk_class"] == "speculative"
        frame.loc[spec, "allocation_aud"] = frame.loc[spec, "allocation_aud"].clip(upper=speculative_cap)
        remainder = investable - frame["allocation_aud"].sum()
        if remainder > 0:
            non_spec = frame[~spec]
            if not non_spec.empty:
                frame.loc[non_spec.index, "allocation_aud"] += (
                    remainder * non_spec["priority"] / non_spec["priority"].sum()
                )
            # If every candidate is speculative the remainder is deliberately
            # left unallocated; it falls through to the cash line below.
        return frame

    # Every CommSec order pays brokerage plus 0.55% FX, so a small parcel is
    # eaten by fees. Capping the order *count* is not enough: the split is
    # priority-weighted, so the tail orders can still land under the minimum.
    # Drop the smallest offender and re-split until every order clears the bar.
    candidates = split(candidates)
    while min_order_aud > 0 and len(candidates) > 1 and candidates["allocation_aud"].min() < min_order_aud:
        candidates = split(candidates.drop(candidates["allocation_aud"].idxmin()))
    if min_order_aud > 0 and len(candidates) == 1 and candidates["allocation_aud"].iloc[0] < min_order_aud:
        return pd.DataFrame(
            [{
                "ticker": "CASH",
                "name": "Cash reserve",
                "allocation_aud": round(monthly_aud, 2),
                "reason": f"Contribution below the AUD {min_order_aud:,.0f} minimum order size; accumulate and invest next month",
                "risk_class": "cash",
                "agent_score": np.nan,
            }]
        )

    candidates["reason"] = candidates.apply(
        lambda r: f"Underweight by {r['gap']:.1%}; critic score {r['agent_score']:.0f}", axis=1
    )
    cash = monthly_aud - candidates["allocation_aud"].sum()
    out = candidates[["ticker", "name", "allocation_aud", "reason", "risk_class", "agent_score"]].copy()
    if cash > 1:
        out = pd.concat(
            [out, pd.DataFrame([{"ticker": "CASH", "name": "Cash reserve", "allocation_aud": cash, "reason": f"{regime.name} cash throttle", "risk_class": "cash", "agent_score": np.nan}])],
            ignore_index=True,
        )
    out["allocation_aud"] = out["allocation_aud"].round(2)
    return out.sort_values("allocation_aud", ascending=False)


def sell_flags(portfolio: pd.DataFrame, critic_scores: pd.Series, current_weight: pd.Series) -> pd.DataFrame:
    p = portfolio.copy()
    p["critic_score"] = p["ticker"].map(critic_scores).fillna(50.0)
    p["current_weight"] = current_weight.reindex(p.index).fillna(0.0)
    p["action"] = "HOLD"
    p["sell_reason"] = "No thesis-break signal"
    low = p["critic_score"] < 35
    p.loc[low, ["action", "sell_reason"]] = ["REVIEW / REDUCE", "Critic score below 35"]
    over = (p["current_weight"] > p["max_weight"] * 1.25) & (p["critic_score"] < 60)
    p.loc[over, ["action", "sell_reason"]] = ["TRIM", "Position exceeds risk cap and conviction is not high"]

    # A broad index fund has no company thesis to break; it is the diversified
    # anchor of the strategy. Only a genuine breach of its risk cap justifies
    # trimming it, never a low single-name-style score.
    core = p["risk_class"].astype(str).str.lower().eq("core") & p["ticker"].ne("CASH")
    core_ok = core & ~(p["current_weight"] > p["max_weight"] * 1.25)
    p.loc[core_ok, ["action", "sell_reason"]] = ["HOLD", "Core index sleeve: hold through drawdowns"]
    p.loc[core & (p["current_weight"] > p["max_weight"] * 1.25), ["action", "sell_reason"]] = [
        "TRIM",
        "Core sleeve exceeds its risk cap",
    ]

    p.loc[p["ticker"] == "CASH", ["action", "sell_reason"]] = ["HOLD", "Liquidity reserve"]
    return p[["ticker", "action", "sell_reason", "critic_score", "current_weight", "max_weight"]]
