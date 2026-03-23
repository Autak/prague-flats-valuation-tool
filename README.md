# Sreality Prague Flats Scraper 🏠

A production-ready Python scraper that pulls **every flat-for-sale listing in Prague** from Sreality.cz's internal JSON API and stores it in a local PostgreSQL database with 50+ ML-ready features per listing.

---

## Quick start

### 1. Prerequisites
- Python 3.11+
- PostgreSQL running locally (e.g. via Postgres.app, Homebrew, or Docker)

### 2. Create the database
```sql
-- run in psql or pgAdmin
CREATE DATABASE sreality;
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure credentials
```bash
cp .env.example .env
# Edit .env → set DB_PASSWORD (and other values if non-default)
```

### 5. Run the scraper
```bash
# Full scrape of all Prague flat listings (~5 000–7 000 listings, ~20–30 min)
python scraper.py

# Test with first 5 pages only
python scraper.py --pages 5

# Tune concurrency and politeness
python scraper.py --workers 6 --delay 1.0
```

### 6. Schedule daily updates
```bash
# Runs immediately, then every 24 hours (keeps the dataset fresh)
python scheduler.py

# Every 12 hours
python scheduler.py --interval 12
```

---

## Database schema — `flats` table

| Column | Type | Description |
|---|---|---|
| `hash_id` | BIGINT PK | Sreality's unique listing ID |
| `listing_url` | TEXT | Direct link to the listing |
| `scraped_at` | TIMESTAMP | First time this listing was captured |
| `updated_at` | TIMESTAMP | Last time this row was refreshed |
| **Price** | | |
| `price` | BIGINT | Asking price in CZK |
| `price_per_m2` | NUMERIC | CZK per m² (computed) |
| `price_note` | TEXT | e.g. "Cena včetně provize" |
| **Core** | | |
| `disposition` | TEXT | 1+kk, 2+1, 3+kk … |
| `usable_area` | NUMERIC | Usable floor area m² |
| `floor_area` | NUMERIC | Total floor area m² |
| **Location** | | |
| `district` | TEXT | Praha 1 … Praha 10, Praha-západ … |
| `neighborhood` | TEXT | Žižkov, Vinohrady, Smíchov … |
| `street` | TEXT | Street name |
| `latitude` | NUMERIC | GPS latitude |
| `longitude` | NUMERIC | GPS longitude |
| `prague_district_num` | INT | 1–10 (parsed from district) |
| **Building** | | |
| `floor` | INT | Floor number |
| `floors_total` | INT | Total floors in building |
| `building_type` | TEXT | Cihlová / Panelová / … |
| `building_condition` | TEXT | Novostavba / Po rekonstrukci / … |
| `energy_class` | TEXT | A – G |
| `year_built` | INT | Year of construction |
| `year_reconstructed` | INT | Year of last reconstruction |
| `lift` | BOOLEAN | Elevator present |
| `cellar` | BOOLEAN | Cellar/storage unit |
| `garage` | BOOLEAN | Garage |
| `parking` | BOOLEAN | Parking space |
| `balcony` | BOOLEAN | Balcony |
| `terrace` | BOOLEAN | Terrace |
| `loggia` | BOOLEAN | Loggia |
| **Ownership** | | |
| `ownership_type` | TEXT | Osobní / Družstevní / Státní |
| `encumbrance` | TEXT | Legal encumbrances |
| **Equipment** | | |
| `furnished` | TEXT | Vybavený / Částečně / Nevybavený |
| `equipment_details` | TEXT[] | Array of individual items |
| **Utilities** | | |
| `heating` | TEXT | Type of heating |
| `water` | TEXT | Water supply type |
| `sewage` | TEXT | Sewage type |
| `gas` | BOOLEAN | Gas connection |
| `telco` | TEXT[] | Telecom/internet options |
| **Photos** | | |
| `photo_count` | INT | Number of photos (quality signal) |
| `main_photo_url` | TEXT | URL of the cover photo |
| **Seller** | | |
| `seller_name` | TEXT | Agent or owner name |
| `seller_type` | TEXT | "agency" or "private" |
| `seller_id` | BIGINT | Sreality seller ID |
| **Meta** | | |
| `is_new_listing` | BOOLEAN | Listed today |
| `tags` | TEXT[] | Sreality labels (Nová cena, …) |

---

## Useful SQL queries

```sql
-- How many listings per district?
SELECT district, COUNT(*) AS n, ROUND(AVG(price)) AS avg_price_czk
FROM flats
GROUP BY district ORDER BY avg_price_czk DESC;

-- Price per m² heatmap data
SELECT latitude, longitude, price_per_m2
FROM flats
WHERE price_per_m2 BETWEEN 50000 AND 300000
  AND latitude IS NOT NULL;

-- Cheapest 2+kk with lift and balcony
SELECT title, price, usable_area, district, listing_url
FROM flats
WHERE disposition = '2+kk'
  AND lift = TRUE
  AND balcony = TRUE
ORDER BY price ASC
LIMIT 20;

-- Feature completeness check (useful before ML)
SELECT
  COUNT(*)                                        AS total,
  COUNT(price)                                    AS has_price,
  COUNT(usable_area)                              AS has_area,
  COUNT(floor)                                    AS has_floor,
  COUNT(building_type)                            AS has_btype,
  COUNT(energy_class)                             AS has_energy,
  COUNT(latitude)                                 AS has_geo
FROM flats;
```

---

## Next steps: Dashboard & ML model

After collecting 1 000+ rows:

1. **Dashboard** – connect PostgreSQL to **Metabase** (free, local) or **Grafana** for instant charts: price distribution, map heatmap, price-per-m² by district, listing volume over time.

2. **ML price prediction** – recommended pipeline:
   - Features: `usable_area`, `floor`, `floors_total`, `prague_district_num`, `building_type`, `energy_class`, boolean amenities, `year_built`, `ownership_type`, `furnished`
   - Target: `price` (log-transform recommended)
   - Models to try: **XGBoost / LightGBM** (best for tabular), then SHAP for explainability
   - Libraries: `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `pandas`

---

## Notes & ethics
- Sreality is operated by Seznam.cz. Their ToS prohibits bulk commercial reuse of listings.
- This scraper is intended for **personal research and model training** only.
- Includes respectful rate-limiting (≥1 s between list pages, randomised jitter on detail calls).
- Run `--delay 2.0` for extra caution.
