# -*- coding: utf-8 -*-
"""
migrate_supabase.py
-------------------
Migrates local PostgreSQL sreality database to Supabase.

Usage:
    python migrate_supabase.py            # full migration
    python migrate_supabase.py --dry-run  # inspect local data, no push
    python migrate_supabase.py --append   # append instead of replace tables
"""

import argparse
import sys
import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ── tqdm is optional (falls back to simple print) ──────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

load_dotenv()

# ── CLI args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Migrate local sreality DB → Supabase")
parser.add_argument("--dry-run", action="store_true", help="Load local data and show stats; do not push to Supabase")
parser.add_argument("--append", action="store_true", help="Append to existing Supabase tables instead of replacing them")
args = parser.parse_args()

CHUNK_SIZE = 500
IF_EXISTS = "append" if args.append else "replace"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Connect to local PostgreSQL and load tables
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — Loading data from local PostgreSQL")
print("=" * 60)

local_db_url = (
    f"postgresql+psycopg2://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','')}"
    f"@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME','sreality')}"
)
try:
    local_engine = create_engine(local_db_url)
    with local_engine.connect() as test_conn:
        test_conn.execute(text("SELECT 1"))
except Exception as e:
    print(f"\n[ERROR] Could not connect to local PostgreSQL: {e}")
    print("   Make sure your local DB is running and .env credentials are correct.")
    sys.exit(1)

with local_engine.connect() as lc:
    df_flats = pd.read_sql("SELECT * FROM flats", lc)
    df_rentals = pd.read_sql("SELECT * FROM rentals", lc)
local_engine.dispose()

local_flats_count = len(df_flats)
local_rentals_count = len(df_rentals)

print(f"  [OK] flats:   {local_flats_count:,} rows loaded")
print(f"  [OK] rentals: {local_rentals_count:,} rows loaded")
print(f"       flats columns:   {list(df_flats.columns)}")
print(f"       rentals columns: {list(df_rentals.columns)}")

# ── Dry-run exits here ──────────────────────────────────────────────────────
if args.dry_run:
    print("\n" + "=" * 60)
    print("DRY RUN -- no data pushed to Supabase")
    print("=" * 60)
    print("\n  flats sample:")
    print(df_flats[["hash_id", "title", "price", "district"]].head(3).to_string(index=False))
    print("\n  rentals sample:")
    print(df_rentals[["hash_id", "title", "monthly_rent", "district"]].head(3).to_string(index=False))
    print("\n[OK] Dry run complete. Re-run without --dry-run to push data.")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Connect to Supabase via SQLAlchemy 2.0
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Connecting to Supabase")
print("=" * 60)

supa_url = os.getenv("SUPABASE_DB_URL")
if not supa_url:
    print("[ERROR] SUPABASE_DB_URL not set in .env")
    sys.exit(1)

try:
    engine = create_engine(
        supa_url,
        connect_args={"sslmode": "require"},
        pool_pre_ping=True,
    )
    # Quick connectivity test
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("  [OK] Supabase connection successful")
except Exception as e:
    print(f"\n[ERROR] Could not connect to Supabase: {e}")
    print("   Check SUPABASE_DB_URL in your .env and ensure the Supabase pooler is reachable.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: push a DataFrame in chunks with progress
# ─────────────────────────────────────────────────────────────────────────────
def push_dataframe(df: pd.DataFrame, table_name: str, if_exists: str):
    """Push df to Supabase via SQLAlchemy 2.0 with chunked inserts."""
    total = len(df)
    print(f"\n  >> Pushing '{table_name}' ({total:,} rows, mode={if_exists}, chunk={CHUNK_SIZE})")

    # SQLAlchemy 2.0: must use `with engine.begin() as conn`
    # First chunk (or replace): create/replace the table structure
    first_chunk = True
    chunks = [df.iloc[i:i + CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)]

    iterator = tqdm(chunks, unit="chunk", desc=f"  {table_name}") if HAS_TQDM else chunks

    for chunk in iterator:
        chunk_if_exists = if_exists if first_chunk else "append"
        with engine.begin() as conn:
            chunk.to_sql(
                table_name,
                con=conn,
                if_exists=chunk_if_exists,
                index=False,
                method="multi",
            )
        first_chunk = False

        if not HAS_TQDM:
            rows_done = min(chunks.index(chunk) * CHUNK_SIZE + len(chunk), total)  # type: ignore
            print(f"    {rows_done:,}/{total:,} rows pushed...")

    print(f"  [OK] '{table_name}' push complete")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Push tables
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 3 — Migrating tables (if_exists='{IF_EXISTS}')")
print("=" * 60)

push_dataframe(df_flats, "flats", IF_EXISTS)
push_dataframe(df_rentals, "rentals", IF_EXISTS)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Verify row counts on Supabase
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Verifying row counts on Supabase")
print("=" * 60)

all_ok = True
with engine.connect() as conn:
    for table, local_count in [("flats", local_flats_count), ("rentals", local_rentals_count)]:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        cloud_count = result.scalar()
        match = cloud_count == local_count
        status = "[OK]" if match else "[MISMATCH]"
        if not match:
            all_ok = False
        print(f"  {status} {table}: local={local_count:,}  supabase={cloud_count:,}")

print("\n" + "=" * 60)
if all_ok:
    print("Migration completed successfully! All row counts match.")
else:
    print("[WARNING] Migration finished but row counts do not match. Check the Supabase logs.")
print("=" * 60 + "\n")
