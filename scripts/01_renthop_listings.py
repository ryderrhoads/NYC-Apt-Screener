#!/usr/bin/env python3
"""
RentHop detail-page scraper — stage 2.

Reads listings from stage 1 JSONL, fetches each detail page, and extracts
features not present on the search cards.

Changes in this version:
  - Reuses one Page per worker instead of opening a new page for every URL
  - Removes the global cooldown lock that stalled all workers at once
  - Uses per-worker backoff on 403s instead
  - Warms each worker session before scraping
  - Randomizes user agent / viewport slightly across contexts
  - Avoids repeated BeautifulSoup scans where easy
  - Uses slower, safer default pacing for detail pages

Usage:
    pip install playwright beautifulsoup4
    playwright install chromium

    python scripts/01_renthop_listings.py \
        --in data/listings.jsonl \
        --out data/details.jsonl \
        --concurrency 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

LISTING_ID_RE = re.compile(r"/(\d{6,})(?:[/?#]|$)")
BATH_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s+Bath(?:room)?s?\b", re.I)
POSTED_RE = re.compile(r"Posted\s+([^,]+?)(?:,|$)", re.I)
DIST_RE = re.compile(r"([\d.]+)\s*mi.*?(\d+)\s*min", re.I)
PRICECOMP_RE = re.compile(
    r"([\d.]+)%\s+(cheaper|more expensive).*?\$([\d,]+)\s+for\s+a\s+"
    r"([\w\s\-]+?)\s+apartment\s+in\s+([^.]+)",
    re.I | re.S,
)

USER_AGENTS = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
]

VIEWPORTS = [
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1280, "height": 800},
]


def _text(n: Optional[Tag]) -> Optional[str]:
    return n.get_text(" ", strip=True) if n else None


def _to_int(s) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return int(s)
    digits = re.sub(r"[^\d]", "", str(s))
    return int(digits) if digits else None


def _to_float(s) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_jsonld(soup: BeautifulSoup) -> dict:
    """Pull the Apartment mainEntity out of the page's JSON-LD block."""
    for s in soup.find_all("script", type="application/ld+json"):
        raw = s.string or s.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("@type") == "Apartment":
                return item
            me = item.get("mainEntity")
            if isinstance(me, dict) and me.get("@type") == "Apartment":
                return me
    return {}


def parse_amenities(soup: BeautifulSoup) -> list[str]:
    """Find AMENITIES header, then grab col-* items from the next flex block."""
    header = soup.find(string=re.compile(r"^\s*AMENITIES\s*$"))
    if not header:
        return []
    container = header.find_next(class_="d-flex")
    if not container:
        return []
    items: list[str] = []
    for d in container.select("div[class*='col-']"):
        t = d.get_text(strip=True)
        if t:
            items.append(t)
    return items


def parse_transit(soup: BeautifulSoup) -> list[dict]:
    """Nearby subway stops with station name, lines, distance, walk time."""
    stops = []
    for stop in soup.select("#nearby-transit .d-block.mt-3"):
        lines = [l.get_text(strip=True) for l in stop.select(".transit-nyc")]
        name_el = stop.select_one("span.b")
        if not name_el:
            continue
        dist_text = None
        for span in stop.find_all("span"):
            t = span.get_text()
            if t and "mi" in t and "min" in t:
                dist_text = t
                break
        mi, mn = None, None
        if dist_text:
            m = DIST_RE.search(dist_text)
            if m:
                mi, mn = float(m.group(1)), int(m.group(2))
        stops.append(
            {
                "station": name_el.get_text(strip=True),
                "lines": lines,
                "distance_mi": mi,
                "walk_min": mn,
            }
        )
    return stops


def parse_price_comparison(soup: BeautifulSoup) -> dict:
    header = soup.find(string=re.compile(r"^\s*PRICE\s+COMPARISON\s*$"))
    if not header:
        return {}
    para = header.find_next(class_="font-size-9")
    if not para:
        return {}
    txt = para.get_text(" ", strip=True)
    m = PRICECOMP_RE.search(txt)
    if not m:
        return {}
    pct = float(m.group(1))
    if m.group(2).lower().startswith("cheap"):
        pct = -pct
    return {
        "pct_vs_area_median": pct,
        "area_median_price": _to_int(m.group(3)),
        "comparison_bed_type": m.group(4).strip().lower(),
        "comparison_area": m.group(5).strip(),
    }


def parse_quality(soup: BeautifulSoup) -> dict:
    header = soup.find(string=re.compile(r"^\s*LISTING\s+QUALITY\s*$"))
    if not header:
        return {"quality_score": None, "quality_items": []}
    table = header.find_next("table")
    if not table:
        return {"quality_score": None, "quality_items": []}
    positives = table.select("td.font-light-green")
    items = [td.get_text(strip=True) for td in table.select("td.font-size-9")]
    return {"quality_score": len(positives), "quality_items": items}


def parse_tags(soup: BeautifulSoup) -> dict:
    tags = set()
    for chip in soup.select(".font-size-8.b, .font-size-8.mr-1.b"):
        t = chip.get_text(strip=True)
        if t in {"No Fee", "Exclusive", "Featured", "By Owner"}:
            tags.add(t)
    return {
        "is_no_fee": "No Fee" in tags,
        "is_exclusive": "Exclusive" in tags,
        "is_featured": "Featured" in tags,
        "is_by_owner": "By Owner" in tags,
        "tags": sorted(tags),
    }


def parse_broker(soup: BeautifulSoup) -> dict:
    verified = (
        soup.find("img", title=re.compile(r"verifying their identity", re.I))
        is not None
    )
    name = None
    company = None
    name_el = soup.select_one(".agent-name a.b")
    if name_el:
        name = name_el.get_text(strip=True)
        parent = name_el.find_parent()
        if parent:
            sibling = parent.find_next_sibling("div")
            if sibling:
                company = sibling.get_text(strip=True)
    return {
        "broker_name": name,
        "broker_company": company,
        "broker_verified": verified,
    }


def parse_bathrooms(soup: BeautifulSoup) -> Optional[float]:
    img = soup.find("img", alt="bathrooms")
    if not img:
        return None
    for sib in img.find_all_next(limit=5):
        t = sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else ""
        m = BATH_RE.search(t)
        if m:
            return float(m.group(1))
    return None


def parse_posted_and_move_in(soup: BeautifulSoup) -> dict:
    out = {"posted": None, "move_in": None}
    for div in soup.find_all("div", class_="font-size-9"):
        t = div.get_text(" ", strip=True)
        if t.startswith("Posted"):
            m = POSTED_RE.search(t)
            if m:
                out["posted"] = m.group(1).strip()
            if "," in t:
                out["move_in"] = t.split(",", 1)[1].strip()
            break
    return out


def parse_num_photos(soup: BeautifulSoup) -> Optional[int]:
    photos = soup.select(".carousel-item.photo-item img.carousel-item-photo")
    return len(photos) or None


def parse_breadcrumb_neighborhoods(soup: BeautifulSoup) -> list[str]:
    items = soup.select(".Breadcrumb .Breadcrumb-item")
    return [i.get_text(strip=True) for i in items if i.get_text(strip=True)]


def parse_detail(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    jl = parse_jsonld(soup)

    lid_m = LISTING_ID_RE.search(url)
    listing_id = lid_m.group(1) if lid_m else None
    if not listing_id and jl.get("@id"):
        lid_m = LISTING_ID_RE.search(jl["@id"])
        listing_id = lid_m.group(1) if lid_m else None

    addr = jl.get("address", {}) if isinstance(jl.get("address"), dict) else {}
    geo = jl.get("geo", {}) if isinstance(jl.get("geo"), dict) else {}

    offer = {}
    for s in soup.find_all("script", type="application/ld+json"):
        raw = s.string or s.get_text(strip=True)
        if not raw:
            continue
        try:
            blob = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for item in (blob if isinstance(blob, list) else [blob]):
            if not isinstance(item, dict):
                continue
            about = item.get("about")
            if isinstance(about, dict):
                o = about.get("offers")
                if isinstance(o, dict):
                    offer = o
                    break
        if offer:
            break

    amenities = parse_amenities(soup)
    transit = parse_transit(soup)
    pricecomp = parse_price_comparison(soup)
    quality = parse_quality(soup)
    tags = parse_tags(soup)
    broker = parse_broker(soup)
    posted = parse_posted_and_move_in(soup)

    nearest = transit[0] if transit else {}

    description = jl.get("description") or ""
    if not description:
        hd = soup.find(string=re.compile(r"^\s*DESCRIPTION\s*$"))
        if hd:
            d = hd.find_next(class_="font-size-9")
            description = d.get_text(" ", strip=True) if d else ""

    return {
        "listing_id": listing_id,
        "url": url,
        "title": _text(soup.find("h1")),
        "street_address": addr.get("streetAddress"),
        "city": addr.get("addressLocality"),
        "state": addr.get("addressRegion"),
        "zip": addr.get("postalCode"),
        "latitude": _to_float(geo.get("latitude")),
        "longitude": _to_float(geo.get("longitude")),
        "price_usd": _to_int(offer.get("price")),
        "bedrooms": (
            _to_int(jl.get("numberOfRooms"))
            if jl.get("numberOfRooms") is not None
            else None
        ),
        "bathrooms": parse_bathrooms(soup),
        "description": description,
        "amenities": amenities,
        "num_amenities": len(amenities),
        "num_photos": parse_num_photos(soup),
        "breadcrumbs": parse_breadcrumb_neighborhoods(soup),
        "transit": transit,
        "num_transit_stops": len(transit),
        "nearest_transit_station": nearest.get("station"),
        "nearest_transit_distance_mi": nearest.get("distance_mi"),
        "nearest_transit_walk_min": nearest.get("walk_min"),
        "nearest_transit_lines": nearest.get("lines"),
        "area_median_price": pricecomp.get("area_median_price"),
        "pct_vs_area_median": pricecomp.get("pct_vs_area_median"),
        "comparison_bed_type": pricecomp.get("comparison_bed_type"),
        "comparison_area": pricecomp.get("comparison_area"),
        "quality_score": quality["quality_score"],
        "quality_items": quality["quality_items"],
        "posted": posted["posted"],
        "move_in": posted["move_in"],
        **tags,
        **broker,
    }


# ---------------------------------------------------------------------------
# Scraper scaffolding
# ---------------------------------------------------------------------------

BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}
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
    "getclicky.com",
    "quantserve.com",
    "criteo",
    "adnxs",
    "adsystem",
    "taboola",
    "outbrain",
)


async def _route_blocker(route: Route) -> None:
    req = route.request
    if req.resource_type in BLOCKED_RESOURCE_TYPES:
        return await route.abort()
    if any(h in req.url for h in BLOCKED_HOSTS):
        return await route.abort()
    await route.continue_()


@dataclass
class Stats:
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    started: float = field(default_factory=time.time)


@dataclass
class WorkerState:
    idx: int
    context: BrowserContext
    page: Page
    backoff_until: float = 0.0
    consecutive_403s: int = 0

    async def wait_until_ready(self) -> None:
        while True:
            now = time.time()
            if now >= self.backoff_until:
                return
            await asyncio.sleep(self.backoff_until - now)

    def push_backoff(self, seconds: float) -> None:
        target = time.time() + seconds
        if target > self.backoff_until:
            self.backoff_until = target


async def _warm_context(page: Page, log: logging.Logger, idx: int) -> None:
    try:
        await page.goto(
            "https://www.renthop.com",
            wait_until="domcontentloaded",
            timeout=20000,
        )
        await asyncio.sleep(random.uniform(2.0, 4.0))
        log.info("worker=%d warmed session", idx)
    except Exception as e:  # noqa: BLE001
        log.warning("worker=%d warmup failed: %s", idx, e)


async def _fetch(
    worker: WorkerState,
    url: str,
    *,
    timeout_ms: int,
) -> tuple[Optional[str], Optional[int]]:
    page = worker.page
    resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    if resp is None:
        return None, None

    status = resp.status
    if not resp.ok:
        return None, status

    try:
        await page.wait_for_selector(
            'script[type="application/ld+json"]',
            timeout=4000,
        )
    except PWTimeoutError:
        return None, status

    return await page.content(), status


async def _scrape_one(
    worker: WorkerState,
    url: str,
    *,
    retries: int,
    timeout_ms: int,
    min_delay: float,
    max_delay: float,
    log: logging.Logger,
) -> Optional[dict]:
    await worker.wait_until_ready()
    await asyncio.sleep(random.uniform(min_delay, max_delay))

    for attempt in range(1, retries + 2):
        try:
            html, status = await _fetch(worker, url, timeout_ms=timeout_ms)
            if html:
                worker.consecutive_403s = 0
                row = parse_detail(html, url)
                log.info(
                    "worker=%d %-70s ok (attempt %d)",
                    worker.idx,
                    url[-70:],
                    attempt,
                )
                return row

            if status == 403:
                worker.consecutive_403s += 1
                cooldown = min(
                    15 * worker.consecutive_403s + random.uniform(3, 8),
                    90,
                )
                worker.push_backoff(cooldown)
                log.warning(
                    "worker=%d 403 on %s attempt=%d; backing off %.1fs",
                    worker.idx,
                    url,
                    attempt,
                    cooldown,
                )
                await worker.wait_until_ready()
                continue

            log.warning(
                "worker=%d %s: status=%s attempt=%d",
                worker.idx,
                url,
                status,
                attempt,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "worker=%d %s: %s (attempt %d)",
                worker.idx,
                url,
                e,
                attempt,
            )

        await asyncio.sleep((2 ** attempt) + random.uniform(0.5, 1.5))

    log.error("worker=%d gave up on %s", worker.idx, url)
    return None


def load_urls(in_path: Path) -> list[tuple[str, str]]:
    """Read stage-1 JSONL, return (listing_id, url) pairs."""
    out = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = row.get("url")
            lid = str(row.get("listing_id", ""))
            if url and lid:
                out.append((lid, url))
    return out


def load_done(out_path: Path) -> set[str]:
    """Read existing output JSONL and return already-scraped listing_ids."""
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                lid = row.get("listing_id")
                if lid:
                    done.add(str(lid))
            except json.JSONDecodeError:
                continue
    return done


async def _worker_loop(
    worker: WorkerState,
    queue: asyncio.Queue[Optional[tuple[str, str]]],
    out_handle,
    stats: Stats,
    stats_lock: asyncio.Lock,
    out_lock: asyncio.Lock,
    *,
    retries: int,
    timeout_ms: int,
    min_delay: float,
    max_delay: float,
    log: logging.Logger,
    progress_every: int,
    total: int,
) -> None:
    processed = 0

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        lid, url = item
        row = await _scrape_one(
            worker,
            url,
            retries=retries,
            timeout_ms=timeout_ms,
            min_delay=min_delay,
            max_delay=max_delay,
            log=log,
        )

        async with stats_lock:
            processed = stats.ok + stats.failed + 1
            if row is None:
                stats.failed += 1
            else:
                stats.ok += 1

            done = stats.ok + stats.failed
            if done % progress_every == 0 or done == total:
                elapsed = time.time() - stats.started
                rate = done / max(elapsed, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                log.info(
                    "progress: %d/%d (ok=%d fail=%d) %.2f/s eta=%ds",
                    done,
                    total,
                    stats.ok,
                    stats.failed,
                    rate,
                    int(eta),
                )

        if row is not None:
            async with out_lock:
                out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_handle.flush()

        queue.task_done()


async def scrape(
    urls: list[tuple[str, str]],
    *,
    concurrency: int,
    retries: int,
    timeout_ms: int,
    min_delay: float,
    max_delay: float,
    headless: bool,
    out_path: Path,
    log: logging.Logger,
    progress_every: int = 25,
) -> Stats:
    stats = Stats()
    queue: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue()
    stats_lock = asyncio.Lock()
    out_lock = asyncio.Lock()

    for item in urls:
        await queue.put(item)
    for _ in range(concurrency):
        await queue.put(None)

    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )

        workers: list[WorkerState] = []
        try:
            for i in range(concurrency):
                context = await browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    viewport=random.choice(VIEWPORTS),
                    java_script_enabled=True,
                )
                page = await context.new_page()
                await page.route("**/*", _route_blocker)
                worker = WorkerState(idx=i, context=context, page=page)
                await _warm_context(page, log, i)
                workers.append(worker)

            with out_path.open("a", encoding="utf-8") as jf:
                tasks = [
                    asyncio.create_task(
                        _worker_loop(
                            worker,
                            queue,
                            jf,
                            stats,
                            stats_lock,
                            out_lock,
                            retries=retries,
                            timeout_ms=timeout_ms,
                            min_delay=min_delay,
                            max_delay=max_delay,
                            log=log,
                            progress_every=progress_every,
                            total=len(urls),
                        )
                    )
                    for worker in workers
                ]

                await queue.join()

                for task in tasks:
                    await task

        finally:
            for worker in workers:
                try:
                    await worker.page.close()
                except Exception:
                    pass
                try:
                    await worker.context.close()
                except Exception:
                    pass
            await browser.close()

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/listings.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/details.jsonl"))
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout-ms", type=int, default=25000)
    ap.add_argument("--min-delay", type=float, default=2.0)
    ap.add_argument("--max-delay", type=float, default=5.0)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap URLs fetched this run (useful for testing)",
    )
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("detail")

    if not args.inp.exists():
        log.error("input file not found: %s", args.inp)
        log.error(
            "run stage 1 first, or pass the correct path with --in data/listings.jsonl"
        )
        return 1

    all_urls = load_urls(args.inp)
    done_ids = load_done(args.out)
    pending = [(lid, url) for lid, url in all_urls if lid not in done_ids]

    seen = set()
    unique = []
    for lid, url in pending:
        if lid in seen:
            continue
        seen.add(lid)
        unique.append((lid, url))
    pending = unique

    if args.limit:
        pending = pending[: args.limit]

    log.info(
        "inputs: %d urls total, %d already done, %d to fetch this run",
        len(all_urls),
        len(done_ids),
        len(pending),
    )

    if not pending:
        log.info("nothing to do")
        return 0

    stats = asyncio.run(
        scrape(
            pending,
            concurrency=args.concurrency,
            retries=args.retries,
            timeout_ms=args.timeout_ms,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            headless=not args.headed,
            out_path=args.out,
            log=log,
        )
    )

    elapsed = time.time() - stats.started
    log.info(
        "done: ok=%d failed=%d in %.1fs (%.2f/s)",
        stats.ok,
        stats.failed,
        elapsed,
        (stats.ok + stats.failed) / max(elapsed, 1e-6),
    )
    return 0 if stats.ok else 1


if __name__ == "__main__":
    sys.exit(main())