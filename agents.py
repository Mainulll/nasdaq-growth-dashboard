from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # optional fallback
    SentimentIntensityAnalyzer = None


@dataclass
class AgentDecision:
    score: float
    label: str
    reasons: list[str]
    warnings: list[str]


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _pct_score(value: Any, bands: list[tuple[float, float]]) -> float:
    if not _finite(value):
        return 0.0
    v = float(value)
    for threshold, score in bands:
        if v >= threshold:
            return score
    return 0.0


def is_fund(row: pd.Series, risk_class: str = "") -> bool:
    """Identify pooled vehicles, which carry no company fundamentals."""
    quote_type = str(row.get("quote_type") or "").upper()
    if quote_type in {"ETF", "MUTUALFUND", "INDEX"}:
        return True
    return str(risk_class).strip().lower() == "core"


def score_etf(row: pd.Series, momentum_6m: float | None = None, momentum_12m: float | None = None) -> AgentDecision:
    """Score a fund on the evidence a fund actually has.

    Running the operating-company model over an ETF scored every fund near zero,
    because Yahoo reports no revenue, margin, cash-flow or leverage data for a
    pooled vehicle. That starved the core sleeve of contributions and flagged it
    for sale. Cost, scale and realised long-run return are the durable evidence
    for an index fund, so those drive the score instead.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    score = 0.0

    # Cost compounds with certainty; it is the most reliable ETF discriminator.
    expense = row.get("expense_ratio")
    if _finite(expense):
        e = float(expense)
        score += 30 if e <= 0.10 else 25 if e <= 0.20 else 20 if e <= 0.40 else 12 if e <= 0.75 else 5
        reasons.append(f"Expense ratio {e:.2f}%")
        if e > 0.75:
            warnings.append("Expense ratio is high for an index fund")
    else:
        score += 18
        warnings.append("Expense ratio unavailable")

    # Scale proxies bid/ask cost and closure risk.
    assets = row.get("net_assets")
    if _finite(assets):
        a = float(assets)
        score += 20 if a >= 1e10 else 16 if a >= 1e9 else 10 if a >= 2e8 else 4
        reasons.append(f"Net assets USD {a/1e9:.1f}B")
        if a < 2e8:
            warnings.append("Small fund: closure and spread risk")
    else:
        score += 10

    for label, key, weight in (("3y", "return_3y", 20), ("5y", "return_5y", 15)):
        value = row.get(key)
        if _finite(value):
            v = float(value)
            score += weight if v >= 0.15 else weight * 0.8 if v >= 0.10 else weight * 0.6 if v >= 0.05 else weight * 0.35 if v >= 0 else 0
            reasons.append(f"{label} average return {v:.1%}")
        else:
            score += weight * 0.5

    beta = row.get("beta_3y")
    if _finite(beta):
        b = float(beta)
        score += 5 if b <= 1.5 else 3 if b <= 2.0 else 1
        if b > 1.5:
            warnings.append(f"Fund beta {b:.2f} amplifies market drawdowns")

    for period, value in (("6m", momentum_6m), ("12m", momentum_12m)):
        if value is not None and _finite(value):
            x = float(value)
            score += 5 if x > 0.15 else 4 if x > 0 else 2 if x > -0.20 else 0
            reasons.append(f"{period} momentum {x:.1%}")

    score = float(np.clip(score, 0, 100))
    label = "High conviction" if score >= 75 else "Positive" if score >= 60 else "Watch" if score >= 45 else "Avoid / thesis review"
    return AgentDecision(score=score, label=label, reasons=reasons[:6], warnings=warnings[:6])


def score_holding(
    row: pd.Series,
    momentum_6m: float | None = None,
    momentum_12m: float | None = None,
    risk_class: str = "",
) -> AgentDecision:
    """Route each holding to the model that matches what it is."""
    if is_fund(row, risk_class):
        return score_etf(row, momentum_6m, momentum_12m)
    return score_fundamental(row, momentum_6m, momentum_12m)


def score_fundamental(row: pd.Series, momentum_6m: float | None = None, momentum_12m: float | None = None) -> AgentDecision:
    reasons: list[str] = []
    warnings: list[str] = []
    score = 0.0

    rev = row.get("revenue_growth")
    rev_score = _pct_score(rev, [(0.40, 20), (0.25, 17), (0.15, 14), (0.08, 10), (0.0, 5), (-999, 0)])
    score += rev_score
    if _finite(rev):
        reasons.append(f"Revenue growth {float(rev):.1%}")
        if float(rev) < 0:
            warnings.append("Revenue is contracting")

    eg = row.get("earnings_growth")
    earn_score = _pct_score(eg, [(0.40, 15), (0.20, 12), (0.10, 9), (0.0, 5), (-999, 0)])
    score += earn_score
    if _finite(eg):
        reasons.append(f"Earnings growth {float(eg):.1%}")

    gm = row.get("gross_margin")
    score += _pct_score(gm, [(0.70, 10), (0.50, 8), (0.35, 6), (0.20, 3), (-999, 0)])
    if _finite(gm) and float(gm) >= 0.50:
        reasons.append("Strong gross margin")

    om = row.get("operating_margin")
    score += _pct_score(om, [(0.30, 10), (0.20, 8), (0.10, 6), (0.0, 3), (-999, 0)])
    if _finite(om) and float(om) < 0:
        warnings.append("Operating margin is negative")

    fcf = row.get("free_cashflow")
    revenue = row.get("total_revenue")
    fcf_margin = np.nan
    if _finite(fcf) and _finite(revenue) and float(revenue) != 0:
        fcf_margin = float(fcf) / float(revenue)
        score += _pct_score(fcf_margin, [(0.25, 10), (0.15, 8), (0.08, 6), (0.0, 3), (-999, 0)])
        if fcf_margin < 0:
            warnings.append("Free cash flow is negative")
        else:
            reasons.append(f"FCF margin {fcf_margin:.1%}")

    de = row.get("debt_to_equity")
    if _finite(de):
        de_value = float(de)
        score += 10 if de_value < 40 else 7 if de_value < 100 else 4 if de_value < 200 else 1
        if de_value > 200:
            warnings.append("High debt-to-equity")
    else:
        score += 4

    cr = row.get("current_ratio")
    if _finite(cr):
        score += 5 if float(cr) >= 1.5 else 3 if float(cr) >= 1.0 else 1
        if float(cr) < 1.0:
            warnings.append("Current ratio below 1")
    else:
        score += 2

    fpe = row.get("forward_pe")
    ps = row.get("price_to_sales")
    valuation = 0.0
    if _finite(fpe) and float(fpe) > 0:
        v = float(fpe)
        valuation += 10 if v <= 25 else 8 if v <= 40 else 5 if v <= 60 else 2 if v <= 100 else 0
        if v > 60:
            warnings.append("Forward P/E embeds high expectations")
    elif _finite(ps):
        v = float(ps)
        valuation += 8 if v <= 5 else 5 if v <= 10 else 2 if v <= 20 else 0
        if v > 15:
            warnings.append("Price-to-sales is elevated")
    score += valuation

    beta = row.get("beta")
    if _finite(beta):
        b = float(beta)
        score += 5 if 0.8 <= b <= 1.8 else 3 if b <= 2.5 else 1
        if b > 2.0:
            warnings.append("Very high beta")
    else:
        score += 2

    for period, value in (("6m", momentum_6m), ("12m", momentum_12m)):
        if value is not None and _finite(value):
            x = float(value)
            add = 5 if x > 0.30 else 4 if x > 0.15 else 3 if x > 0 else 1 if x > -0.20 else 0
            score += add
            reasons.append(f"{period} momentum {x:.1%}")

    score = float(np.clip(score, 0, 100))
    label = "High conviction" if score >= 75 else "Positive" if score >= 60 else "Watch" if score >= 45 else "Avoid / thesis review"
    return AgentDecision(score=score, label=label, reasons=reasons[:6], warnings=warnings[:6])


def portfolio_critic(
    row: pd.Series,
    fundamental: AgentDecision,
    current_weight: float,
    target_weight: float,
    max_weight: float,
    portfolio_sector_weight: float,
    sleeve: str,
) -> AgentDecision:
    score = fundamental.score
    reasons = list(fundamental.reasons)
    warnings = list(fundamental.warnings)

    if current_weight > max_weight:
        penalty = min(20.0, (current_weight - max_weight) * 200)
        score -= penalty
        warnings.append(f"Position exceeds {max_weight:.0%} cap")

    if portfolio_sector_weight > 0.35:
        score -= 10
        warnings.append("Portfolio has excessive exposure to this sector/theme")

    if sleeve.lower().startswith("spec"):
        score -= 8
        warnings.append("Speculative sleeve: execution and financing risk dominate")

    fcf = row.get("free_cashflow")
    om = row.get("operating_margin")
    if _finite(fcf) and float(fcf) < 0 and _finite(om) and float(om) < 0:
        score -= 10
        warnings.append("Negative operating margin and free cash flow")

    score = float(np.clip(score, 0, 100))
    label = "Add" if score >= 75 and current_weight < target_weight else "Hold / add selectively" if score >= 60 else "Hold / monitor" if score >= 45 else "Reduce / avoid"
    return AgentDecision(score=score, label=label, reasons=reasons[:6], warnings=warnings[:8])


_POSITIVE = {"beat", "surge", "record", "growth", "approval", "contract", "partnership", "upgrade", "profit", "launch", "expansion"}
_NEGATIVE = {"miss", "lawsuit", "probe", "downgrade", "delay", "loss", "recall", "ban", "fraud", "default", "cut", "warning"}
_SENSATIONAL = {"soars", "plunges", "explodes", "crashes", "unstoppable", "disaster", "panic", "guaranteed"}


def headline_sentiment(text: str) -> float:
    if not text:
        return 0.0
    if SentimentIntensityAnalyzer is not None:
        try:
            return float(SentimentIntensityAnalyzer().polarity_scores(text)["compound"])
        except Exception:
            pass
    tokens = set(re.findall(r"[A-Za-z]+", text.lower()))
    raw = len(tokens & _POSITIVE) - len(tokens & _NEGATIVE)
    return float(np.clip(raw / 3, -1, 1))


def analyse_news(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return news
    out = news.copy()
    out["sentiment"] = out["headline"].fillna("").map(headline_sentiment)
    out["sensational"] = out["headline"].fillna("").str.lower().apply(lambda x: any(w in x for w in _SENSATIONAL))
    out["duplicate_key"] = out["headline"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.slice(0, 80)
    return out.drop_duplicates("duplicate_key")


def news_critic(news: pd.DataFrame) -> AgentDecision:
    if news.empty:
        return AgentDecision(25, "Low confidence", [], ["No current news data available"])
    source_share = news["source"].fillna("unknown").value_counts(normalize=True)
    dominant = float(source_share.iloc[0]) if len(source_share) else 1.0
    sensational_share = float(news.get("sensational", pd.Series(False, index=news.index)).mean())
    dispersion = float(news.get("sentiment", pd.Series(0.0, index=news.index)).std(ddof=0))
    score = 80.0
    warnings: list[str] = []
    reasons: list[str] = [f"{news['source'].nunique()} distinct sources", f"{len(news)} unique headlines"]
    if dominant > 0.50:
        score -= 25
        warnings.append("News set is dominated by one source")
    if sensational_share > 0.20:
        score -= 20
        warnings.append("High share of sensational headlines")
    if dispersion > 0.55:
        score -= 10
        warnings.append("Narrative is highly polarised")
    if len(news) < 5:
        score -= 20
        warnings.append("Evidence set is too small")
    label = "Reliable enough for context" if score >= 70 else "Use cautiously" if score >= 45 else "Do not trade from this news set"
    return AgentDecision(float(np.clip(score, 0, 100)), label, reasons, warnings)
