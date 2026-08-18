"""
Live market data — stocks via yfinance (no key needed), crypto via
CoinGecko's free public API (no key needed).

Note: yfinance's `fast_info` field names have shifted across versions in
the past. If an attribute below raises, run `dir(ticker.fast_info)` to see
what your installed version actually exposes.
"""
import requests
import yfinance as yf


def get_stock_quote(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.fast_info
    return {
        "text": (
            f"{ticker.upper()} last price: {info.last_price:.2f} "
            f"{info.currency}. Day range: {info.day_low:.2f}-{info.day_high:.2f}. "
            f"Previous close: {info.previous_close:.2f}."
        ),
        "source": f"Yahoo Finance ({ticker.upper()})",
    }


def get_crypto_price(coin_id: str = "bitcoin", vs_currency: str = "usd") -> dict:
    resp = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": coin_id,
            "vs_currencies": vs_currency,
            "include_24hr_change": "true",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get(coin_id, {})
    change = data.get(f"{vs_currency}_24h_change", 0) or 0
    return {
        "text": (
            f"{coin_id.capitalize()} price: {data.get(vs_currency)} "
            f"{vs_currency.upper()} (24h change: {change:.2f}%)."
        ),
        "source": "CoinGecko",
    }
