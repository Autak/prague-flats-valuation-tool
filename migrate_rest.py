import pandas as pd
import psycopg2
import os
import math
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# LOCAL CONN
local_conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432'),
    dbname=os.getenv('DB_NAME', 'sreality'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', '')
)

df_sales = pd.read_sql("SELECT * FROM flats", local_conn)
df_rentals = pd.read_sql("SELECT * FROM rentals", local_conn)
local_conn.close()

# SUPABASE CLIENT (REST API wrapper over port 443)
url: str = 'https://ipmcnhuncxsybbtzgbfd.supabase.co'
key: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlwbWNuaHVuY3hzeWJidHpnYmZkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ5NDU1MjAsImV4cCI6MjA5MDUyMTUyMH0.f2or2BjD-Mlv1oHFGV7LKPrXu8oOXrL28VeCwLjtrt4'
supabase: Client = create_client(url, key)

def push_table(df, table_name):
    import json
    records = json.loads(df.to_json(orient='records', date_format='iso'))
    
    chunk_size = 500
    chunk_size = 500
    total = len(records)
    print(f"Pushing {total} rows to {table_name} via REST API...")
    
    for i in range(0, total, chunk_size):
        chunk = records[i:i+chunk_size]
        response = supabase.table(table_name).insert(chunk).execute()
        print(f"Migrated {min(i+chunk_size, total)}/{total}")

push_table(df_sales, "flats")
push_table(df_rentals, "rentals")

print("Migration successful! Verifying count...")
res_flats = supabase.table("flats").select("hash_id", count="exact").execute()
print("Flats row count on Supabase:", res_flats.count)

res_rentals = supabase.table("rentals").select("hash_id", count="exact").execute()
print("Rentals row count on Supabase:", res_rentals.count)
