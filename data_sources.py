from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Bounded fan-out: enough to hide per-request latency, low enough to stay
# well inside Yahoo/Finnhub free-tier rate limits from a shared cloud IP.
_MAX_WORKERS = 8


@dataclass
class DataStatus:
    source: str
    delayed: bool
    message: str


def _clean_tickers(tickers: Iterable[str]) -> list[str]:
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip() and str(t).upper() != "CASH"})


def download_prices(tickers: Iterable[str], period: str = "1y", interval: str = "1d") -> tuple[pd.DataFrame, DataStatus]:
    symbols = _clean_tickers(tickers)
    if not symbols:
        return pd.DataFrame(), DataStatus("none", True, "No market tickers supplied")
    try:
        raw = yf.download(
            symbols,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
        if raw.empty:
            raise RuntimeError("No price data returned")
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            close = raw[["Close"]].rename(columns={"Close": symbols[0]})
        close = close.dropna(how="all")
        return close, DataStatus("Yahoo Finance via yfinance", True, "Suitable for research; quotes may be delayed or adjusted")
    except Exception as exc:
        return pd.DataFrame(), DataStatus("Yahoo Finance via yfinance", True, f"Price download failed: {exc}")


def finnhub_quotes(tickers: Iterable[str], api_key: str | None) -> tuple[pd.DataFrame, DataStatus]:
    symbols = _clean_tickers(tickers)
    if not api_key:
        return pd.DataFrame(), DataStatus("Finnhub", False, "No FINNHUB_API_KEY configured")
    def fetch(symbol: str) -> dict[str, Any]:
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": symbol, "token": api_key},
                timeout=12,
            )
            r.raise_for_status()
            q = r.json()
            # Finnhub answers unknown or unentitled symbols with a zero-filled
            # payload rather than an error. Treating that 0 as a price would
            # silently value the position at nothing, so drop it to NaN.
            price = _safe_num(q.get("c"))
            if not math.isfinite(price) or price <= 0:
                return {"ticker": symbol}
            return {
                "ticker": symbol,
                "price": price,
                "change": _safe_num(q.get("d")),
                "change_pct": _safe_num(q.get("dp")),
                "high": _safe_num(q.get("h")),
                "low": _safe_num(q.get("l")),
                "open": _safe_num(q.get("o")),
                "previous_close": _safe_num(q.get("pc")),
                "timestamp": q.get("t"),
            }
        except Exception:
            return {"ticker": symbol}

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(symbols))) as pool:
        rows = list(pool.map(fetch, symbols))

    out = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
    priced = int(out["price"].notna().sum()) if "price" in out.columns else 0
    if priced == 0:
        return pd.DataFrame(), DataStatus("Finnhub", False, "Finnhub returned no usable quotes; using delayed prices")
    return out, DataStatus(
        "Finnhub",
        False,
        f"Free personal-use quote feed; {priced}/{len(symbols)} symbols priced. Entitlements and delays may vary",
    )


def _safe_num(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def ticker_fundamentals(symbol: str) -> dict[str, Any]:
    """Fetch a compact fundamental record. Uses yfinance as a convenience layer.

    The dashboard explicitly treats this as research data and preserves SEC links for verification.
    """
    record: dict[str, Any] = {"ticker": symbol}
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        record.update(
            {
                "name": info.get("shortName") or info.get("longName") or symbol,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": _safe_num(info.get("marketCap")),
                "enterprise_value": _safe_num(info.get("enterpriseValue")),
                "trailing_pe": _safe_num(info.get("trailingPE")),
                "forward_pe": _safe_num(info.get("forwardPE")),
                "price_to_sales": _safe_num(info.get("priceToSalesTrailing12Months")),
                "ev_to_ebitda": _safe_num(info.get("enterpriseToEbitda")),
                "revenue_growth": _safe_num(info.get("revenueGrowth")),
                "earnings_growth": _safe_num(info.get("earningsGrowth")),
                "gross_margin": _safe_num(info.get("grossMargins")),
                "operating_margin": _safe_num(info.get("operatingMargins")),
                "profit_margin": _safe_num(info.get("profitMargins")),
                "free_cashflow": _safe_num(info.get("freeCashflow")),
                "operating_cashflow": _safe_num(info.get("operatingCashflow")),
                "total_revenue": _safe_num(info.get("totalRevenue")),
                "total_cash": _safe_num(info.get("totalCash")),
                "total_debt": _safe_num(info.get("totalDebt")),
                "debt_to_equity": _safe_num(info.get("debtToEquity")),
                "current_ratio": _safe_num(info.get("currentRatio")),
                "beta": _safe_num(info.get("beta")),
                "recommendation": info.get("recommendationKey"),
                "target_mean_price": _safe_num(info.get("targetMeanPrice")),
                "currency": info.get("currency", "USD"),
                "quote_type": info.get("quoteType"),
                # Fund-only fields. Yahoo returns none of the operating-company
                # metrics above for an ETF, so without these a fund has no
                # scoreable evidence at all.
                "expense_ratio": _safe_num(info.get("netExpenseRatio")),
                "net_assets": _safe_num(info.get("netAssets") or info.get("totalAssets")),
                "fund_category": info.get("category"),
                "fund_family": info.get("fundFamily"),
                "beta_3y": _safe_num(info.get("beta3Year")),
                "return_3y": _safe_num(info.get("threeYearAverageReturn")),
                "return_5y": _safe_num(info.get("fiveYearAverageReturn")),
            }
        )
    except Exception as exc:
        record["error"] = str(exc)
    return record


def fundamentals_table(tickers: Iterable[str]) -> pd.DataFrame:
    symbols = _clean_tickers(tickers)
    if not symbols:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(symbols))) as pool:
        rows = list(pool.map(ticker_fundamentals, symbols))
    return pd.DataFrame(rows).set_index("ticker")


def gdelt_news(query: str, days: int = 7, max_records: int = 50) -> pd.DataFrame:
    try:
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": max_records,
            "format": "json",
            "timespan": f"{max(1, days)}d",
            "sort": "datedesc",
        }
        r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
        rows = []
        for article in payload.get("articles", []):
            rows.append(
                {
                    "headline": article.get("title"),
                    "source": article.get("domain"),
                    "url": article.get("url"),
                    "published": article.get("seendate"),
                    "language": article.get("language"),
                    "country": article.get("sourcecountry"),
                    "image": article.get("socialimage"),
                }
            )
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def finnhub_company_news(symbol: str, api_key: str | None, days: int = 7) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    end = date.today()
    start = end - timedelta(days=days)
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": symbol, "from": start.isoformat(), "to": end.isoformat(), "token": api_key},
            timeout=15,
        )
        r.raise_for_status()
        rows = []
        for item in r.json():
            ts = item.get("datetime")
            rows.append(
                {
                    "ticker": symbol,
                    "headline": item.get("headline"),
                    "summary": item.get("summary"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "published": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
                    "category": item.get("category"),
                }
            )
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def fred_series(series_id: str, api_key: str | None, limit: int = 400) -> pd.Series:
    if not api_key:
        return pd.Series(dtype=float, name=series_id)
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            },
            timeout=15,
        )
        r.raise_for_status()
        observations = r.json().get("observations", [])
        data = {
            pd.Timestamp(x["date"]): float(x["value"])
            for x in observations
            if x.get("value") not in (None, ".")
        }
        return pd.Series(data, name=series_id).sort_index()
    except Exception:
        return pd.Series(dtype=float, name=series_id)


FX_FALLBACK_AUDUSD = 0.65


def current_audusd() -> tuple[float, DataStatus]:
    """Return the live AUD/USD rate and an explicit status.

    `FastInfo.get()` silently returns None for keys that `__getitem__` resolves,
    so a `.get()` lookup here degraded to the hard-coded fallback without ever
    raising. Every AUD conversion in the app was then wrong by whatever the rate
    had drifted, with the dashboard still presenting it as live. Index access is
    the only reliable read, and the caller is told when the fallback is in use.
    """
    for source in ("AUDUSD=X", "AUDUSD%3DX"):
        try:
            fast = yf.Ticker(source).fast_info
            raw = fast["last_price"]
            rate = float(raw)
            if math.isfinite(rate) and rate > 0:
                return rate, DataStatus("Yahoo Finance FX", True, f"AUD/USD {rate:.4f} live")
        except Exception:
            continue
    try:
        hist = yf.Ticker("AUDUSD=X").history(period="5d")["Close"].dropna()
        if not hist.empty:
            rate = float(hist.iloc[-1])
            if math.isfinite(rate) and rate > 0:
                return rate, DataStatus("Yahoo Finance FX", True, f"AUD/USD {rate:.4f} from recent close")
    except Exception:
        pass
    return FX_FALLBACK_AUDUSD, DataStatus(
        "Fallback constant",
        True,
        f"FX lookup failed — using stale fallback {FX_FALLBACK_AUDUSD:.4f}. AUD figures are unreliable.",
    )
