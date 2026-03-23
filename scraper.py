"""
Sreality.cz Prague Flats Scraper
=================================
Fetches flat-for-sale listings in Prague via the internal Sreality JSON API,
enriches each listing with its detail page, and upserts everything into
a local PostgreSQL database.

Usage
-----
    python scraper.py [--pages N] [--workers N] [--delay SECONDS]

Requirements
------------
    pip install requests psycopg2-binary python-dotenv tqdm

Set your DB credentials in .env (see README.md).
"""

import os
import time
import random
import logging
import argparse
from datetime import datetime
from typing import Optional

import requests
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "sreality"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# Sreality internal API – no auth required, returns JSON
LIST_API  = "https://www.sreality.cz/api/cs/v2/estates"
DETAIL_API = "https://www.sreality.cz/api/cs/v2/estates/{hash_id}"

# Praha = category_main=1 (byty/flats), category_type=1 (prodej/sale)
# locality_region_id=10  → Hlavní město Praha
BASE_PARAMS = {
    "category_main_cb":   1,   # byty (flats)
    "category_type_cb":   1,   # prodej (sale)
    "locality_region_id": 10,  # Praha
    "per_page":           60,  # max items per page
    "tms":                int(time.time()),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "cs-CZ,cs;q=0.9,en;q=0.8",
    "Referer":         "https://www.sreality.cz/",
    "Origin":          "https://www.sreality.cz",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ──────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_db(conn):
    """Create the flats table if it doesn't exist yet."""
    ddl = """
    CREATE TABLE IF NOT EXISTS flats (
        -- identity
        hash_id             BIGINT PRIMARY KEY,
        listing_url         TEXT,
        scraped_at          TIMESTAMP DEFAULT NOW(),
        updated_at          TIMESTAMP DEFAULT NOW(),

        -- price
        price               BIGINT,
        price_per_m2        NUMERIC(10,2),
        price_note          TEXT,
        currency            TEXT DEFAULT 'CZK',

        -- core attributes
        title               TEXT,
        description         TEXT,
        disposition         TEXT,         -- 1+kk, 2+1, …
        usable_area         NUMERIC(8,2), -- m²
        floor_area          NUMERIC(8,2),
        land_area           NUMERIC(8,2),

        -- location
        district            TEXT,         -- Praha 1, Praha 2, …
        municipality        TEXT,
        neighborhood        TEXT,
        street              TEXT,
        address_full        TEXT,
        latitude            NUMERIC(10,7),
        longitude           NUMERIC(10,7),
        prague_district_num INT,          -- 1-10

        -- building
        floor               INT,
        floors_total        INT,
        building_type       TEXT,         -- panel, cihla, …
        building_condition  TEXT,
        energy_class        TEXT,         -- A-G
        year_built          INT,
        year_reconstructed  INT,
        lift                BOOLEAN,
        cellar              BOOLEAN,
        garage              BOOLEAN,
        parking             BOOLEAN,
        balcony             BOOLEAN,
        terrace             BOOLEAN,
        loggia              BOOLEAN,

        -- ownership & legal
        ownership_type      TEXT,         -- osobní, družstevní, státní
        encumbrance         TEXT,

        -- equipment
        furnished           TEXT,         -- vybavený, částečně, nevybavený
        equipment_details   TEXT[],

        -- utilities
        heating             TEXT,
        water               TEXT,
        sewage              TEXT,
        gas                 BOOLEAN,
        telco               TEXT[],

        -- photo
        photo_count         INT,
        main_photo_url      TEXT,

        -- seller
        seller_name         TEXT,
        seller_type         TEXT,         -- private / agency
        seller_id           BIGINT,

        -- meta
        is_new_listing      BOOLEAN,
        tags                TEXT[]
    );

    CREATE INDEX IF NOT EXISTS idx_flats_district     ON flats(district);
    CREATE INDEX IF NOT EXISTS idx_flats_price        ON flats(price);
    CREATE INDEX IF NOT EXISTS idx_flats_disposition  ON flats(disposition);
    CREATE INDEX IF NOT EXISTS idx_flats_area         ON flats(usable_area);
    CREATE INDEX IF NOT EXISTS idx_flats_scraped_at   ON flats(scraped_at);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    log.info("Database schema ready.")


# ──────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────
def _get(url, params=None, retries=4, backoff=3):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = backoff * (2 ** attempt) + random.uniform(0, 2)
                log.warning("Rate-limited. Sleeping %.1fs…", wait)
                time.sleep(wait)
            else:
                log.warning("HTTP %s for %s", r.status_code, url)
                time.sleep(backoff)
        except Exception as exc:
            log.warning("Request error (%s): %s", attempt + 1, exc)
            time.sleep(backoff * (attempt + 1))
    return None


def fetch_listing_page(page: int) -> Optional[dict]:
    params = {**BASE_PARAMS, "page": page, "tms": int(time.time())}
    return _get(LIST_API, params=params)


def fetch_detail(hash_id: int) -> Optional[dict]:
    url = DETAIL_API.format(hash_id=hash_id)
    return _get(url)


# ──────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────
def _val(items: list, name: str):
    """Pull a value from the estate_detail items array by name key."""
    for item in items:
        if item.get("name") == name:
            v = item.get("value")
            if isinstance(v, list):
                return v[0].get("value") if v else None
            return v
    return None


def _int_floor(v):
    """Extract an integer from floor strings like '3. podlaží' or '3'."""
    if v is None:
        return None
    import re
    m = re.match(r'\s*(-?\d+)', str(v))
    return int(m.group(1)) if m else None


def _bval(items, name):
    v = _val(items, name)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("ano", "true", "1", "yes")


def _arr(items, name):
    for item in items:
        if item.get("name") == name:
            v = item.get("value", [])
            if isinstance(v, list):
                return [str(x.get("value", "")) for x in v]
    return []


DISPOSITION_MAP = {
    2: "1+kk", 3: "1+1", 4: "2+kk", 5: "2+1",
    6: "3+kk", 7: "3+1", 8: "4+kk", 9: "4+1",
    10: "5+kk", 11: "5+1", 12: "6+kk", 13: "6+1",
    16: "Atypický",
}

BUILDING_TYPE_MAP = {
    1: "Cihlová", 2: "Panelová", 3: "Smíšená", 4: "Skeletová",
    5: "Kamenná", 6: "Dřevostavba", 7: "Jiná",
}

OWNERSHIP_MAP = {
    1: "Osobní", 2: "Družstevní", 3: "Státní/obecní", 4: "Ostatní",
}

CONDITION_MAP = {
    1: "Velmi dobrý", 2: "Dobrý", 3: "Špatný", 4: "Panel",
    5: "Cihlový", 6: "Novostavba", 7: "Před rekonstrukcí",
    8: "Po rekonstrukci", 9: "Projekt",
}

FURNISHED_MAP = {
    1: "Vybavený", 2: "Částečně vybavený", 3: "Nevybavený",
}


def parse_flat(summary: dict, detail: dict) -> dict:
    """Merge listing-page summary with the full detail response."""
    di = detail.get("items", []) if detail else []

    # --- location ---
    # Sreality changed `locality` to a simple string in the summary object!
    location_str = str(summary.get("locality") or (detail.get("locality", {}).get("value") if detail and isinstance(detail.get("locality"), dict) else detail.get("locality", "")))
    address_full = location_str if location_str else None

    # district e.g. "Praha 3" → 3
    district_raw = None
    prag_num = None
    import re
    m = re.search(r'(Praha\s*(?:\d+|-východ|-západ)?)', location_str)
    if m:
        district_raw = m.group(1).strip()
        num_m = re.search(r'Praha\s*(\d+)', district_raw)
        if num_m:
            prag_num = int(num_m.group(1))

    # --- price ---
    price_raw = summary.get("price") or (detail.get("price_czk", {}).get("value_raw") if detail else None)
    usable    = summary.get("usable_area") or (
        _val(di, "Plocha") or _val(di, "Užitná plocha")
    )

    price_per_m2 = None
    try:
        if price_raw and usable:
            price_per_m2 = round(float(price_raw) / float(usable), 2)
    except (TypeError, ZeroDivisionError):
        pass

    # --- seller ---
    seller = (detail or {}).get("_embedded", {}).get("seller", {})

    # --- photos ---
    images   = (detail or {}).get("_embedded", {}).get("images", [])
    main_img = images[0].get("_links", {}).get("view", {}).get("href") if images else None

    disposition_cb = summary.get("category_sub_cb") or 0

    return {
        "hash_id":            summary.get("hash_id"),
        "listing_url":        f"https://www.sreality.cz/detail/prodej/byt/{disposition_cb}/{summary.get('hash_id')}",
        "scraped_at":         datetime.utcnow(),

        # price
        "price":              price_raw,
        "price_per_m2":       price_per_m2,
        "price_note":         (detail or {}).get("price_czk", {}).get("name"),
        "currency":           "CZK",

        # core
        "title":              summary.get("name") or (detail or {}).get("name", {}).get("value"),
        "description":        (detail or {}).get("text", {}).get("value"),
        "disposition":        DISPOSITION_MAP.get(disposition_cb, str(disposition_cb)),
        "usable_area":        usable,
        "floor_area":         _val(di, "Podlahová plocha"),
        "land_area":          _val(di, "Plocha pozemku"),

        # location
        "district":           district_raw,
        "municipality":       None,
        "neighborhood":       None,
        "street":             None,
        "address_full":       address_full,
        "latitude":           summary.get("gps", {}).get("lat"),
        "longitude":          summary.get("gps", {}).get("lon"),
        "prague_district_num": prag_num,

        # building
        "floor":              _int_floor(_val(di, "Podlaží") or summary.get("floor")),
        "floors_total":       _int_floor(_val(di, "Počet podlaží")),
        "building_type":      BUILDING_TYPE_MAP.get(_val(di, "Stavba"), _val(di, "Stavba")),
        "building_condition": CONDITION_MAP.get(_val(di, "Stav objektu"), _val(di, "Stav objektu")),
        "energy_class":       _val(di, "Energetická náročnost budovy"),
        "year_built":         _int_floor(_val(di, "Rok stavby")),
        "year_reconstructed": _int_floor(_val(di, "Rok rekonstrukce")),
        "lift":               _bval(di, "Výtah"),
        "cellar":             _bval(di, "Sklep"),
        "garage":             _bval(di, "Garáž"),
        "parking":            _bval(di, "Parkování"),
        "balcony":            _bval(di, "Balkon"),
        "terrace":            _bval(di, "Terasa"),
        "loggia":             _bval(di, "Lodžie"),

        # ownership
        "ownership_type":     OWNERSHIP_MAP.get(_val(di, "Vlastnictví"), _val(di, "Vlastnictví")),
        "encumbrance":        _val(di, "Věcné břemeno"),

        # equipment
        "furnished":          FURNISHED_MAP.get(_val(di, "Vybavení"), _val(di, "Vybavení")),
        "equipment_details":  _arr(di, "Vybavení bytu"),

        # utilities
        "heating":            _val(di, "Topení"),
        "water":              _val(di, "Voda"),
        "sewage":             _val(di, "Odpad"),
        "gas":                _bval(di, "Plyn"),
        "telco":              _arr(di, "Telekomunikace"),

        # photos
        "photo_count":        len(images),
        "main_photo_url":     main_img,

        # seller
        "seller_name":        seller.get("user_name") or seller.get("official_name"),
        "seller_type":        "agency" if seller.get("is_agency") else "private",
        "seller_id":          seller.get("user_id"),

        # meta
        "is_new_listing":     summary.get("new_today", False),
        "tags":               [t.get("name") for t in summary.get("labels", []) if t.get("name")],
    }


# ──────────────────────────────────────────────
# Upsert
# ──────────────────────────────────────────────
UPSERT_SQL = """
INSERT INTO flats (
    hash_id, listing_url, scraped_at, updated_at,
    price, price_per_m2, price_note, currency,
    title, description, disposition, usable_area, floor_area, land_area,
    district, municipality, neighborhood, street, address_full,
    latitude, longitude, prague_district_num,
    floor, floors_total, building_type, building_condition,
    energy_class, year_built, year_reconstructed,
    lift, cellar, garage, parking, balcony, terrace, loggia,
    ownership_type, encumbrance, furnished, equipment_details,
    heating, water, sewage, gas, telco,
    photo_count, main_photo_url,
    seller_name, seller_type, seller_id,
    is_new_listing, tags
) VALUES (
    %(hash_id)s, %(listing_url)s, %(scraped_at)s, %(scraped_at)s,
    %(price)s, %(price_per_m2)s, %(price_note)s, %(currency)s,
    %(title)s, %(description)s, %(disposition)s, %(usable_area)s,
    %(floor_area)s, %(land_area)s,
    %(district)s, %(municipality)s, %(neighborhood)s, %(street)s, %(address_full)s,
    %(latitude)s, %(longitude)s, %(prague_district_num)s,
    %(floor)s, %(floors_total)s, %(building_type)s, %(building_condition)s,
    %(energy_class)s, %(year_built)s, %(year_reconstructed)s,
    %(lift)s, %(cellar)s, %(garage)s, %(parking)s,
    %(balcony)s, %(terrace)s, %(loggia)s,
    %(ownership_type)s, %(encumbrance)s, %(furnished)s, %(equipment_details)s,
    %(heating)s, %(water)s, %(sewage)s, %(gas)s, %(telco)s,
    %(photo_count)s, %(main_photo_url)s,
    %(seller_name)s, %(seller_type)s, %(seller_id)s,
    %(is_new_listing)s, %(tags)s
)
ON CONFLICT (hash_id) DO UPDATE SET
    updated_at        = EXCLUDED.scraped_at,
    price             = EXCLUDED.price,
    price_per_m2      = EXCLUDED.price_per_m2,
    price_note        = EXCLUDED.price_note,
    title             = EXCLUDED.title,
    description       = EXCLUDED.description,
    usable_area       = EXCLUDED.usable_area,
    district          = EXCLUDED.district,
    address_full      = EXCLUDED.address_full,
    latitude          = EXCLUDED.latitude,
    longitude         = EXCLUDED.longitude,
    floor             = EXCLUDED.floor,
    building_condition= EXCLUDED.building_condition,
    energy_class      = EXCLUDED.energy_class,
    photo_count       = EXCLUDED.photo_count,
    tags              = EXCLUDED.tags;
"""


def upsert_flats(conn, rows: list[dict]):
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, UPSERT_SQL, rows, page_size=100)
    conn.commit()


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def count_pages() -> int:
    data = fetch_listing_page(1)
    if not data:
        raise RuntimeError("Failed to reach Sreality API – check your internet connection.")
    total = data.get("result_size", 0)
    per   = BASE_PARAMS["per_page"]
    pages = (total + per - 1) // per
    log.info("Total listings: %d → %d pages", total, pages)
    return pages


def scrape(max_pages: int = 0, workers: int = 4, delay: float = 1.2):
    conn = get_connection()
    init_db(conn)

    total_pages = count_pages()
    if max_pages:
        total_pages = min(total_pages, max_pages)

    all_summaries = []

    # ── Step 1: collect all listing summaries (fast) ──
    log.info("Step 1/2 – Fetching %d listing pages…", total_pages)
    for page in tqdm(range(1, total_pages + 1), unit="page"):
        data = fetch_listing_page(page)
        if data:
            estates = data.get("_embedded", {}).get("estates", [])
            all_summaries.extend(estates)
        time.sleep(delay + random.uniform(0, 0.5))

    log.info("Collected %d summaries. Step 2/2 – Fetching details…", len(all_summaries))

    # ── Step 2: fetch detail pages in parallel ──
    batch = []
    failed = 0

    def fetch_and_parse(summary):
        # Do not process unknown prices!
        if summary.get("price", 0) <= 1:
            return None
        hid = summary.get("hash_id")
        detail = fetch_detail(hid)
        time.sleep(random.uniform(0.3, 0.8))
        return parse_flat(summary, detail)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_and_parse, s): s for s in all_summaries}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="flat"):
            try:
                row = fut.result()
                if row:
                    batch.append(row)
                if len(batch) >= 200:
                    upsert_flats(conn, batch)
                    batch.clear()
            except Exception as exc:
                log.warning("Parse error: %s", exc)
                failed += 1

    if batch:
        upsert_flats(conn, batch)

    conn.close()
    log.info(
        "Done. Saved %d listings (%d failed). DB: %s@%s/%s",
        len(all_summaries) - failed, failed,
        DB_CONFIG["user"], DB_CONFIG["host"], DB_CONFIG["dbname"],
    )


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sreality Prague flats scraper")
    parser.add_argument("--pages",   type=int,   default=0,   help="Max pages (0 = all)")
    parser.add_argument("--workers", type=int,   default=4,   help="Parallel detail workers")
    parser.add_argument("--delay",   type=float, default=1.2, help="Seconds between list pages")
    args = parser.parse_args()

    scrape(max_pages=args.pages, workers=args.workers, delay=args.delay)
