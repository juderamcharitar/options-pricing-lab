"""Read-only Alpaca Market Data client for U.S. equity option chains."""

from datetime import date, timedelta
import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .market import MAX_QUOTES, Quote, timestamp


DATA_URL = "https://data.alpaca.markets"
SYMBOL = re.compile(r"[A-Z][A-Z0-9]{0,5}")
OCC_SUFFIX = re.compile(r"(\d{6})([CP])(\d{8})")
MAX_RESPONSE_BYTES = 2_500_000


class ProviderError(RuntimeError):
    """A market data request or its response failed."""


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _get(path, params, headers):
    request = Request(f"{DATA_URL}{path}?{urlencode(params)}", headers=headers)
    try:
        with build_opener(_NoRedirects()).open(request, timeout=8) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderError("Alpaca returned a response that exceeded the size limit.")
        data = json.loads(raw)
    except HTTPError as error:
        messages = {
            401: "Alpaca rejected the credentials. Check the key and secret in the launch shell.",
            403: "Alpaca denied access to this feed. Check your subscription or choose Indicative.",
            404: "Alpaca could not find market data for that symbol.",
            429: "Alpaca rate limit reached. Wait a moment before refreshing.",
        }
        raise ProviderError(messages.get(error.code, f"Alpaca returned HTTP {error.code}.")) from error
    except (URLError, OSError) as error:
        raise ProviderError("Could not reach Alpaca. Check the connection and try again.") from error
    except (ValueError, UnicodeError) as error:
        raise ProviderError("Alpaca returned an unreadable market data response.") from error
    if not isinstance(data, dict):
        raise ProviderError("Alpaca returned an unexpected response.")
    return data


def _provider_price(value):
    if type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 100_000:
        return float(value)
    return None


def _option_quote(symbol, contract, snapshot):
    suffix = OCC_SUFFIX.fullmatch(contract[len(symbol):]) if contract.startswith(symbol) else None
    if not suffix or not isinstance(snapshot, dict):
        return None
    digits, side, strike_digits = suffix.groups()
    try:
        expiry = date(2000 + int(digits[:2]), int(digits[2:4]), int(digits[4:6]))
    except ValueError:
        return None
    strike = int(strike_digits) / 1000
    if strike <= 0:
        return None
    quote = snapshot.get("latestQuote") or {}
    if not isinstance(quote, dict):
        quote = {}
    try:
        quote_time = timestamp(quote["t"]) if quote.get("t") else None
    except ValueError:
        quote_time = None
    return Quote(expiry, strike, "call" if side == "C" else "put",
                 _provider_price(quote.get("bp")), _provider_price(quote.get("ap")), quote_time, contract)


def snapshot(symbol, feed, expiry, as_of):
    if not isinstance(symbol, str) or not SYMBOL.fullmatch(symbol):
        raise ValueError("Use a U.S. stock or ETF symbol of up to six letters or digits.")
    if feed not in ("indicative", "opra"):
        raise ValueError("Choose Indicative or OPRA for options quotes.")
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise ValueError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the shell that launches the lab.")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Accept": "application/json"}
    stock_feed = "iex"
    stock = _get(f"/v2/stocks/{symbol}/trades/latest", {"feed": stock_feed}, headers)
    trade = stock.get("trade")
    spot = _provider_price(trade.get("p")) if isinstance(trade, dict) else None
    if not spot:
        raise ProviderError("No usable underlying trade was returned by the Alpaca IEX feed.")
    try:
        spot_time = timestamp(trade.get("t"))
    except ValueError as error:
        raise ProviderError("Alpaca returned a trade without a usable timestamp.") from error

    params = {"feed": feed, "root_symbol": symbol, "limit": 500,
              "strike_price_gte": spot * .80, "strike_price_lte": spot * 1.20}
    last_expiry = expiry or as_of + timedelta(days=45)
    if expiry is None:
        params.update(expiration_date_gte=(as_of + timedelta(days=1)).isoformat(),
                      expiration_date_lte=last_expiry.isoformat())
    else:
        params["expiration_date"] = expiry.isoformat()
    quotes = {}
    truncated = False
    for _ in range(2):
        page = _get(f"/v1beta1/options/snapshots/{symbol}", params, headers)
        snapshots = page.get("snapshots")
        if not isinstance(snapshots, dict):
            raise ProviderError("Alpaca returned an unexpected option chain.")
        for contract, values in snapshots.items():
            quote = _option_quote(symbol, contract, values)
            if quote and as_of < quote.expiry <= last_expiry and (expiry is None or quote.expiry == expiry):
                quotes[contract] = quote
        token = page.get("next_page_token")
        truncated = bool(token) or len(quotes) > MAX_QUOTES
        if not token or len(quotes) >= MAX_QUOTES:
            break
        if not isinstance(token, str) or len(token) > 4096 or token == params.get("page_token"):
            raise ProviderError("Alpaca returned an invalid continuation token.")
        params["page_token"] = token
    if not quotes:
        raise ProviderError("Alpaca returned no matching contracts. Try another expiry or leave it blank.")
    return {"quotes": list(quotes.values())[:MAX_QUOTES], "spot": spot, "spot_time": spot_time.isoformat(),
            "stock_feed": stock_feed, "truncated": truncated}
