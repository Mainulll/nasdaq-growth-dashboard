# High-Risk Growth Portfolio Dashboard

A deployable Streamlit research dashboard for a 5–10 year, high-risk growth strategy investing **AUD 2,000 per month** through CommSec International.

## What is included

- 16-holding balanced-growth target portfolio plus dynamic cash
- 50% QQQ / 30% SPY / 20% IWO blended benchmark
- live Finnhub quotes and company news when a free API key is supplied
- historical prices and convenience fundamentals through `yfinance`
- deterministic fundamental-analysis agent for operating companies
- separate fund model for ETFs, scored on cost, scale and realised long-run return
- independent portfolio critic that can veto concentration, leverage, valuation and negative-cash-flow risks
- position-level unrealised P&L in AUD, including the currency contribution
- GDELT world/company news analysis with a second source-quality critic
- market-regime cash throttle
- monthly contribution allocator limited to 1–6 orders, with a minimum order size, to control CommSec fee drag
- sell/trim rules based on thesis deterioration—not merely a falling share price
- GitHub Actions jobs for weekday and month-end snapshots

## Entering your holdings

Enter units and average AUD cost per holding in the sidebar, then download the
updated CSV and commit it (or keep it locally). Two conventions matter:

- **CASH row:** put your uninvested AUD balance in the `units` column. Cash is
  counted in the weight denominator, so the regime cash target is only
  meaningful once you fill this in.
- **`avg_cost_aud`:** your average purchase cost per unit in AUD. This drives
  the P&L table; market value converts the USD price at the live AUD/USD rate,
  so reported P&L includes currency movement as well as share-price movement.

## Important limitation

No free source lawfully provides unlimited, consolidated, real-time prices, complete fundamentals and unrestricted news for every Nasdaq and NYSE security. This project therefore uses a layered design:

1. **Finnhub:** live/watchlist quotes and company news on a free personal-use key.
2. **Yahoo Finance through yfinance:** delayed historical prices and convenience fundamentals.
3. **SEC EDGAR:** authoritative filing verification before material decisions.
4. **GDELT:** broad world-event/news discovery.
5. **FRED:** optional macroeconomic series.

The app is research software, not financial advice or an automated broker.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # optional
streamlit run app.py
```

Open the local URL printed by Streamlit. The API keys are optional—without them
the app falls back to delayed yfinance prices and GDELT news. Never commit
`.streamlit/secrets.toml`; it is gitignored.

## Free deployment

1. Create a private GitHub repository and upload this folder.
2. Register free API keys for Finnhub and FRED.
3. In GitHub, add repository secrets named `FINNHUB_API_KEY`, `FRED_API_KEY`, and `SEC_USER_AGENT`.
4. In Streamlit Community Cloud, create an app from the repository with `app.py` as the entrypoint.
5. Add the same values under Streamlit **App settings → Secrets**.
6. Enable GitHub Actions write permissions if the scheduled workflow cannot commit `data/latest_snapshot.json`.

## Known operational risks

- **Yahoo rate limiting.** `yfinance` is an unofficial scraper. Requests from
  shared cloud IPs (Streamlit Cloud, GitHub Actions) are throttled more often
  than from a home connection. The app degrades to whatever it can fetch and
  reports the source in the footer, so check that line before trusting a number.
- **FX fallback.** If the AUD/USD lookup fails entirely the app shows a red
  error banner and every AUD figure should be treated as unreliable until it
  clears.
- **Fundamentals are convenience data.** Verify against SEC EDGAR before acting
  on anything material.

## CommSec execution controls

CommSec currently supports fractional and recurring investments for eligible U.S. securities, but brokerage applies to each recurring order. The app therefore defaults to three or fewer monthly orders and keeps a dynamic cash reserve.

Before every trade:

- verify the live quote and order type in CommSec
- review brokerage and the 0.55% FX conversion cost
- use limit orders for volatile or thinly traded securities
- confirm that the position remains below its maximum portfolio weight
- check the latest SEC filing and company earnings release

## Initial target allocation

The model starts with a 4% cash reserve in a risk-on regime and can increase cash to 40% in a defensive regime. Speculative positions are capped at 3–4% targets, while the core ETF and profitable growth companies carry larger weights.

Edit `portfolio.csv` to change targets, caps or the research universe.
