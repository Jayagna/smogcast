"""Shared HTTP helper: retries with backoff and a polite rate limit."""
from __future__ import annotations

import time

import requests

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "smogcast/0.1 (air-quality research project)"
_last_call = 0.0


def get(url: str, *, params: dict | None = None, headers: dict | None = None,
        min_interval: float = 1.1, retries: int = 3, timeout: int = 45) -> requests.Response:
    """GET with backoff on 408/429/5xx. `min_interval` spaces out calls
    (OpenAQ's free tier allows 60 requests/minute and 2,000/hour)."""
    global _last_call
    last_err = "no response"
    for attempt in range(retries):
        wait = min_interval - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            r = _SESSION.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}"
            time.sleep(5 * (attempt + 1))
            continue
        # back off before we hit the per-minute limit
        remaining = r.headers.get("x-ratelimit-remaining")
        if remaining is not None and remaining.isdigit() and int(remaining) <= 1:
            time.sleep(min(float(r.headers.get("x-ratelimit-reset") or 60), 60))
        if r.status_code in (408, 429) or r.status_code >= 500:
            last_err = f"HTTP {r.status_code}"
            retry_after = r.headers.get("Retry-After") or r.headers.get("x-ratelimit-reset")
            try:
                delay = float(retry_after) if retry_after else 5 * (attempt + 1)
            except ValueError:
                delay = 5 * (attempt + 1)
            time.sleep(min(delay, 60))
            continue
        if r.status_code in (401, 403):
            raise SystemExit(f"HTTP {r.status_code} from {url.split('?')[0]}: check the API key in .env")
        r.raise_for_status()
        return r
    raise RuntimeError(f"{last_err} after {retries} attempts")
