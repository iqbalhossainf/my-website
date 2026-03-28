import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_KEY = os.getenv("COINGECKO_DEMO_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

HEADERS = {}
if COINGECKO_KEY:
    HEADERS["x-cg-demo-api-key"] = COINGECKO_KEY

TICKER_COINS = [
    ("bitcoin", "BTC"),
    ("ethereum", "ETH"),
    ("binancecoin", "BNB"),
    ("solana", "SOL"),
    ("ripple", "XRP"),
    ("cardano", "ADA"),
]

def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def extract_json_block(text: str) -> dict:
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in AI response")

    return json.loads(match.group(0))

def fetch_market_data() -> dict:
    ids = ",".join([coin_id for coin_id, _ in TICKER_COINS])

    price_resp = requests.get(
        f"{COINGECKO_BASE}/simple/price",
        params={
            "ids": ids,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
        },
        headers=HEADERS,
        timeout=30,
    )
    price_resp.raise_for_status()
    price_data = price_resp.json()

    prices = []
    for coin_id, symbol in TICKER_COINS:
        item = price_data.get(coin_id, {})
        prices.append(
            {
                "id": coin_id,
                "symbol": symbol,
                "price": item.get("usd"),
                "change_24h": item.get("usd_24h_change"),
            }
        )

    markets_resp = requests.get(
        f"{COINGECKO_BASE}/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h",
        },
        headers=HEADERS,
        timeout=30,
    )
    markets_resp.raise_for_status()
    markets = markets_resp.json()

    cleaned = []
    for item in markets:
        change = item.get("price_change_percentage_24h_in_currency")
        if change is None:
            continue
        cleaned.append(
            {
                "name": item.get("name"),
                "symbol": str(item.get("symbol", "")).upper(),
                "price": item.get("current_price"),
                "change_24h": change,
            }
        )

    bullish = sorted(cleaned, key=lambda x: x["change_24h"], reverse=True)[:5]
    bearish = sorted(cleaned, key=lambda x: x["change_24h"])[:5]

    return {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "prices": prices,
        "bullish": bullish,
        "bearish": bearish,
    }

def fetch_ai_news() -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        existing = DATA_DIR / "news.json"
        if existing.exists():
            return json.loads(existing.read_text(encoding="utf-8"))
        return {
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "headlines": [],
            "articles": [],
        }

    client = OpenAI(api_key=api_key)

    prompt = """
Find the latest important crypto news from roughly the last 24 hours.

Use web search.
Pick the 8 most useful stories for a crypto media homepage.
Prefer important market, ETF, regulation, exchange, Bitcoin, Ethereum, stablecoin, and major altcoin stories.
Avoid duplicates.
Write short SEO-friendly summaries in simple English.

Return ONLY valid JSON with exactly this structure:
{
  "headlines": [
    "headline 1",
    "headline 2",
    "headline 3",
    "headline 4",
    "headline 5"
  ],
  "articles": [
    {
      "title": "string",
      "summary": "2 sentence summary",
      "source": "publisher name",
      "url": "https://example.com/article",
      "published_at": "YYYY-MM-DD or ISO timestamp if known",
      "category": "Market"
    }
  ]
}
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        tools=[{"type": "web_search"}],
        input=prompt,
    )

    parsed = extract_json_block(response.output_text)

    return {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "headlines": parsed.get("headlines", [])[:5],
        "articles": parsed.get("articles", [])[:8],
    }

def main() -> None:
    market = fetch_market_data()
    news = fetch_ai_news()

    save_json(DATA_DIR / "market.json", market)
    save_json(DATA_DIR / "news.json", news)

if __name__ == "__main__":
    main()
