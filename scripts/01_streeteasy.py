from patchright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import csv
import time
import random
import sys
from pathlib import Path


class StreetEasyScraper:
    def __init__(self, headless=True):
        self.headless = headless
        self.results = []
        self.seen_urls = set()
        Path("data").mkdir(exist_ok=True)
        Path("debug").mkdir(exist_ok=True)
        # Persistent profile so cookies/PX tokens survive between runs
        self.profile_dir = Path(".browser_profile").resolve()
        self.profile_dir.mkdir(exist_ok=True)

    # ---------- timing ----------

    def _sleep(self, lo=4, hi=10):
        if random.random() < 0.15:
            t = random.uniform(20, 45)
        else:
            t = random.uniform(lo, hi)
        time.sleep(t)

    def _human_scroll(self, page):
        """Scroll in chunks to mimic reading and trigger lazy content."""
        try:
            for _ in range(random.randint(3, 6)):
                page.mouse.wheel(0, random.randint(300, 700))
                time.sleep(random.uniform(0.3, 0.9))
        except Exception as e:
            print(f"  scroll failed (page likely closed): {e}")
            raise  # let run() handle recovery

    # ---------- parsing ----------

    def parse_next_data(self, html):
        soup = BeautifulSoup(html, "lxml")
        tag = soup.find("script", {"id": "__NEXT_DATA__"})
        if not tag:
            return None
        try:
            return json.loads(tag.string)
        except Exception:
            return None

    def parse_jsonld(self, html):
        soup = BeautifulSoup(html, "lxml")
        scripts = soup.find_all("script", {"type": "application/ld+json"})
        items = []
        for s in scripts:
            try:
                d = json.loads(s.string)
            except Exception:
                continue
            if isinstance(d, dict):
                items.extend(d.get("@graph", []))
        return [i for i in items if i.get("@type") == "Apartment"]

    def normalize(self, item):
        addr = item.get("address", {})
        geo = item.get("geo", {})
        props = item.get("additionalProperty", [])
        price = "NA"
        for p in props:
            if p.get("name") == "Monthly Rent":
                price = p.get("value")
        return {
            "type": "rental",
            "neighborhood": addr.get("addressLocality", "NA"),
            "address": item.get("name", "NA"),
            "price": price,
            "geo": f"{geo.get('latitude')},{geo.get('longitude')}",
            "beds": item.get("numberOfBedrooms", "NA"),
            "baths": item.get("numberOfBathroomsTotal", "NA"),
            "sqft": item.get("floorSize", {}).get("value", "NA"),
            "url": item.get("url", "NA"),
        }

    def harvest(self, html, page_num):
        """Extract new unique listings. Dump __NEXT_DATA__ for inspection."""
        nd = self.parse_next_data(html)
        if nd:
            with open(f"debug/next_data_page{page_num}.json", "w") as f:
                json.dump(nd, f, indent=2)

        added = 0
        for item in self.parse_jsonld(html):
            url = item.get("url", "")
            if not url or url in self.seen_urls:
                continue
            self.seen_urls.add(url)
            self.results.append(self.normalize(item))
            added += 1
        return added

    # ---------- output ----------

    def save(self):
        if not self.results:
            print("  nothing to save yet")
            return
        with open("data/streeteasy.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.results[0].keys())
            w.writeheader()
            w.writerows(self.results)

    def get_next_page_url(self, html):
        soup = BeautifulSoup(html, "lxml")

        # 1. Try rel=next (rare but clean)
        link = soup.find("link", {"rel": "next"})
        if link and link.get("href"):
            return "https://streeteasy.com" + link["href"]

        # 2. Correct selector (your case)
        btn = soup.find("a", {"aria-labelledby": "next-arrow-label"})
        if btn and btn.get("href"):
            return "https://streeteasy.com" + btn["href"]

        # 3. Backup: class-based (more brittle but useful)
        btn = soup.select_one("a.NavigationArrow_arrowLink__jfaTM")
        if btn and btn.get("href"):
            return "https://streeteasy.com" + btn["href"]

        return None
    # ---------- driver ----------

    def run(self, max_pages=50):
        base_url = (
            "https://streeteasy.com/for-rent/nyc/"
            "price:-4000%7Carea:101,123,140,313,373,401,402,409,414,415,416,"
            "417,418,419,420,428,431,451,453,454,455,459%7Cbeds:1%7C"
            "in_rect:40.728,40.776,-73.881,-73.795%7C"
            "amenities:elevator,parking,doorman%7Cpets:allowed"
            "?sort_by=price_asc"
        )

        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=self.headless,
                viewport={"width": 1440, "height": 900},
                locale="en-US",
                timezone_id="America/New_York",
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                args=["--disable-blink-features=AutomationControlled"],
            )
            page = context.pages[0] if context.pages else context.new_page()

            # Warmup — land on homepage, scroll, then navigate deeper.
            print("Warmup...")
            try:
                resp = page.goto("https://streeteasy.com/",
                                 wait_until="domcontentloaded", timeout=30000)
                print(f"  homepage: {resp.status if resp else '?'}")
                if resp and resp.status == 403:
                    print("Blocked on homepage. Wait it out or change IP.")
                    context.close()
                    return
            except Exception as e:
                print(f"  homepage nav failed: {e}")
                context.close()
                return

            self._sleep(3, 6)
            try:
                self._human_scroll(page)
            except Exception:
                pass

            try:
                page.goto("https://streeteasy.com/for-rent/nyc",
                          wait_until="domcontentloaded", timeout=30000)
                self._sleep(3, 6)
                self._human_scroll(page)
            except Exception as e:
                print(f"  rentals landing nav/scroll failed: {e} — continuing anyway")
            url= base_url
            empty_streak = 0
            for pnum in range(1, max_pages + 1):
                next_url = self.get_next_page_url(html)
                if not next_url:
                    print("No next page found — stopping")
                    break

                url = next_url
                print(f"Page {pnum}:")

                html = None
                try:
                    resp = page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    status = resp.status if resp else None
                    print(f"  status: {status}")
                    if status == 403:
                        print("  blocked mid-scrape — stopping")
                        break

                    self._sleep(2, 4)
                    self._human_scroll(page)
                    html = page.content()

                except Exception as e:
                    print(f"  page {pnum} failed: {e}")
                    # Recover with a fresh tab in the same context
                    try:
                        page.close()
                    except Exception:
                        pass
                    try:
                        page = context.new_page()
                        print("  opened fresh page, continuing to next iteration")
                    except Exception as e2:
                        print(f"  couldn't recover context: {e2} — stopping")
                        break
                    continue

                if html is None:
                    continue

                with open(f"debug/page{pnum}.html", "w") as f:
                    f.write(html)

                added = self.harvest(html, pnum)
                print(f"  +{added} new (total unique: {len(self.results)})")

                # Save every page — crashes are cheap now
                self.save()

                if added == 0:
                    empty_streak += 1
                    if empty_streak >= 2:
                        print("Two pages with 0 new listings — stopping")
                        break
                else:
                    empty_streak = 0

                self._sleep(5, 12)

            try:
                context.close()
            except Exception:
                pass

        self.save()
        print(f"Done. Saved {len(self.results)} unique listings to data/streeteasy.csv")


if __name__ == "__main__":
    headed = "--headed" in sys.argv
    StreetEasyScraper(headless=not headed).run()