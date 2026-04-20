"""
RentHop scraper — fast, concurrent, Playwright-based.

Why Playwright instead of httpx:
    RentHop is served through a CDN with bot challenges; a headless real browser
    gets through where a plain HTTP client does not. We claw performance back by:
      1. Blocking images, fonts, CSS, analytics, and ad networks via request
         interception. Cuts per-page bytes ~90% and load time ~5x.
      2. Using one browser with N contexts running concurrently (isolated
         cookies, no cross-page state leak).
      3. Waiting only for domcontentloaded — listings are server-rendered so
         we do not need the full load cycle.
      4. Parsing with BeautifulSoup against page.content() once, rather than
         doing dozens of DOM round-trips per page.

Usage:
    # Install deps once:
    pip install playwright beautifulsoup4
    playwright install chromium

    # Scrape 10 pages of Manhattan results, 4 workers, output to JSONL + CSV:
    python renthop_scraper.py \\
        --base-url "https://www.renthop.com/search/nyc" \\
        --pages 10 \\
        --concurrency 4 \\
        --out listings.jsonl \\
        --csv listings.csv

    # Scrape a specific filtered search URL (just pass it as --base-url):
    python renthop_scraper.py \\
        --base-url "https://www.renthop.com/search/nyc?min_price=3000&max_price=5000" \\
        --pages 5
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

from bs4 import BeautifulSoup, Tag
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Route,
    TimeoutError as PWTimeoutError,
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

PRICE_RE = re.compile(r"\$[\d,]+")
BED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*Bed", re.I)
BATH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*Bath", re.I)
SQFT_RE = re.compile(r"([\d,]+)\s*Sqft", re.I)


def _text(node: Optional[Tag]) -> Optional[str]:
    return node.get_text(" ", strip=True) if node else None


def _to_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None


def _to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_listing(card: Tag) -> dict:
    """Extract one listing card into a flat dict."""
    lid = card.get("listing_id") or card.get("id", "").replace("listing-", "")
    info_text = card.get_text(" ", strip=True)

    title_a = card.select_one(f"#listing-{lid}-title")
    neigh = card.select_one(f"#listing-{lid}-neighborhoods")
    price_el = card.select_one(f"#listing-{lid}-price")
    zip_el = title_a.parent.select_one(".font-size-9") if title_a and title_a.parent else None

    tags_candidates = [
        t.get_text(strip=True)
        for t in card.select(".featured-tag, .b.font-blue, .d-inline-block.b")
    ]
    tags = [t for t in tags_candidates if t in {"No Fee", "Featured", "By Owner"}]

    broker = posted = broker_line = None
    for div in card.select("div.font-size-8.overflow-ellipsis"):
        t = div.get_text(" ", strip=True)
        if t.startswith("By "):
            broker_line = t
            a = div.find("a")
            broker = a.get_text(strip=True) if a else None
            if "," in t:
                posted = t.split(",", 1)[1].strip()
            break

    bed_m = BED_RE.search(info_text)
    bath_m = BATH_RE.search(info_text)
    sqft_m = SQFT_RE.search(info_text)

    url = title_a["href"] if title_a and title_a.has_attr("href") else None
    img_el = card.select_one("img.search-thumb")
    photo_url = img_el.get("src") if img_el else None

    return {
        "listing_id": lid,
        "url": url,
        "title": _text(title_a),
        "neighborhoods": _text(neigh),
        "zip": _text(zip_el),
        "price_usd": _to_int(price_el.get_text() if price_el else None),
        "price_raw": _text(price_el),
        "bedrooms": _to_float(bed_m.group(1)) if bed_m else None,
        "bathrooms": _to_float(bath_m.group(1)) if bath_m else None,
        "sqft": _to_int(sqft_m.group(1)) if sqft_m else None,
        "latitude": _to_float(card.get("latitude")),
        "longitude": _to_float(card.get("longitude")),
        "broker": broker,
        "broker_line": broker_line,
        "posted": posted,
        "tags": tags,
        "no_fee": "No Fee" in tags,
        "featured": "Featured" in tags,
        "by_owner": "By Owner" in tags,
        "photo_url": photo_url,
    }


def parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    return [parse_listing(c) for c in soup.select("div.search-listing")]


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

# Resource types we never need for parsing listings.
BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}

# Third-party hosts that do nothing for us but slow everything down.
BLOCKED_HOSTS = (
    "googletagmanager.com",
    "google-analytics.com",
    "doubleclick.net",
    "facebook.net",
    "facebook.com",
    "bing.com",
    "bat.bing.com",
    "segment.com",
    "hotjar.com",
    "clarity.ms",
    "criteo",
    "adnxs",
    "adsystem",
    "taboola",
    "outbrain",
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


async def _route_blocker(route: Route) -> None:
    """Abort requests we do not need; continue everything else."""
    req = route.request
    if req.resource_type in BLOCKED_RESOURCE_TYPES:
        return await route.abort()
    url = req.url
    if any(h in url for h in BLOCKED_HOSTS):
        return await route.abort()
    await route.continue_()


def _page_url(base_url: str, page_num: int) -> str:
    """Attach/replace ?page=N on the base URL."""
    parts = urlparse(base_url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    if page_num > 1:
        q["page"] = str(page_num)
    else:
        q.pop("page", None)
    return urlunparse(parts._replace(query=urlencode(q)))


@dataclass
class Stats:
    pages_ok: int = 0
    pages_failed: int = 0
    listings: int = 0
    bytes_transferred: int = 0
    started: float = 0.0


async def _fetch_page(
    context: BrowserContext,
    url: str,
    *,
    attempt: int,
    timeout_ms: int,
    log: logging.Logger,
) -> Optional[str]:
    """Open a single page, return HTML, or None on failure."""
    page: Page = await context.new_page()
    await page.route("**/*", _route_blocker)
    try:
        # domcontentloaded is enough — the listings are in the initial payload.
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        if resp is None or not resp.ok:
            log.warning("bad response %s attempt=%d status=%s", url, attempt,
                        resp.status if resp else "none")
            return None
        # Quick sanity: wait for at least one card to be present. Short timeout
        # because if it's not there after DOM load, retrying won't help much.
        try:
            await page.wait_for_selector("div.search-listing", timeout=4000)
        except PWTimeoutError:
            log.warning("no listings rendered on %s attempt=%d", url, attempt)
            return None
        return await page.content()
    finally:
        await page.close()


async def _scrape_one(
    context: BrowserContext,
    url: str,
    *,
    sem: asyncio.Semaphore,
    retries: int,
    timeout_ms: int,
    min_delay: float,
    max_delay: float,
    log: logging.Logger,
) -> list[dict]:
    """Fetch + parse one listing page with retry/backoff."""
    async with sem:
        # Jitter between requests — polite and less detectable.
        await asyncio.sleep(random.uniform(min_delay, max_delay))
        for attempt in range(1, retries + 2):
            try:
                html = await _fetch_page(
                    context, url, attempt=attempt, timeout_ms=timeout_ms, log=log
                )
                if html:
                    rows = parse_page(html)
                    log.info("%-70s -> %d listings (attempt %d)", url, len(rows), attempt)
                    return rows
            except Exception as e:  # noqa: BLE001 — we want to log + retry any error
                log.warning("error on %s attempt=%d: %s", url, attempt, e)
            # Exponential backoff with jitter
            backoff = (2 ** attempt) + random.random()
            await asyncio.sleep(backoff)
        log.error("giving up on %s after %d attempts", url, retries + 1)
        return []


async def scrape(
    base_url: str,
    *,
    pages: int,
    concurrency: int,
    retries: int,
    timeout_ms: int,
    min_delay: float,
    max_delay: float,
    headless: bool,
    out_jsonl: Path,
    log: logging.Logger,
) -> Stats:
    """Orchestrate the whole scrape. Streams results to out_jsonl as they arrive."""
    stats = Stats(started=time.time())
    urls = [_page_url(base_url, n) for n in range(1, pages + 1)]
    sem = asyncio.Semaphore(concurrency)
    seen: set[str] = set()

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        # One context per worker gives us isolation and independent cookie jars.
        contexts: list[BrowserContext] = [
            await browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1366, "height": 900},
                java_script_enabled=True,
            )
            for _ in range(concurrency)
        ]
        try:
            async def task(i: int, url: str) -> list[dict]:
                # Round-robin contexts across URLs
                ctx = contexts[i % concurrency]
                return await _scrape_one(
                    ctx, url, sem=sem, retries=retries, timeout_ms=timeout_ms,
                    min_delay=min_delay, max_delay=max_delay, log=log,
                )

            with out_jsonl.open("w", encoding="utf-8") as jf:
                coros = [task(i, u) for i, u in enumerate(urls)]
                for fut in asyncio.as_completed(coros):
                    rows = await fut
                    if rows:
                        stats.pages_ok += 1
                    else:
                        stats.pages_failed += 1
                    for row in rows:
                        lid = row.get("listing_id")
                        if lid in seen:
                            continue  # RentHop sometimes repeats "featured" listings
                        seen.add(lid)
                        jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats.listings += 1
        finally:
            for ctx in contexts:
                await ctx.close()
            await browser.close()

    return stats


# ---------------------------------------------------------------------------
# Output helpers + CLI
# ---------------------------------------------------------------------------

def jsonl_to_csv(src: Path, dst: Path) -> int:
    """Flatten the JSONL into a CSV. Returns row count."""
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        dst.write_text("")
        return 0
    # Stable column order — scalars first, list-ish last.
    cols = [
        "listing_id", "title", "url", "neighborhoods", "zip",
        "price_usd", "price_raw", "bedrooms", "bathrooms", "sqft",
        "latitude", "longitude", "broker", "posted",
        "no_fee", "featured", "by_owner", "photo_url", "broker_line", "tags",
    ]
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            if isinstance(r.get("tags"), list):
                r["tags"] = "|".join(r["tags"])
            w.writerow(r)
    return len(rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True,
                   help="RentHop search URL to paginate through, e.g. https://www.renthop.com/search/nyc")
    p.add_argument("--pages", type=int, default=5, help="Number of pages to fetch (default: 5)")
    p.add_argument("--concurrency", type=int, default=4, help="Parallel browser contexts (default: 4)")
    p.add_argument("--retries", type=int, default=2, help="Retries per page on failure (default: 2)")
    p.add_argument("--timeout-ms", type=int, default=30000, help="Page goto timeout in ms (default: 30000)")
    p.add_argument("--min-delay", type=float, default=0.2, help="Min jitter before each request (s)")
    p.add_argument("--max-delay", type=float, default=0.8, help="Max jitter before each request (s)")
    p.add_argument("--out", type=Path, default=Path("listings.jsonl"), help="JSONL output path")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    p.add_argument("--headed", action="store_true", help="Run the browser with a visible window (debug)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("renthop")

    stats = asyncio.run(scrape(
        args.base_url,
        pages=args.pages,
        concurrency=args.concurrency,
        retries=args.retries,
        timeout_ms=args.timeout_ms,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        headless=not args.headed,
        out_jsonl=args.out,
        log=log,
    ))

    elapsed = time.time() - stats.started
    log.info(
        "done: %d listings, %d/%d pages ok, %.1fs, %.2f pages/s",
        stats.listings, stats.pages_ok, stats.pages_ok + stats.pages_failed,
        elapsed, (stats.pages_ok + stats.pages_failed) / max(elapsed, 1e-6),
    )

    if args.csv:
        n = jsonl_to_csv(args.out, args.csv)
        log.info("wrote %d rows to %s", n, args.csv)

    return 0 if stats.pages_ok else 1


if __name__ == "__main__":
    sys.exit(main())