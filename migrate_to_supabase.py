import pandas as pd
from sqlalchemy import create_engine
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

print("1. Connecting to Local Postgres...")
local_conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432'),
    dbname=os.getenv('DB_NAME', 'sreality'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', '')
)

df_sales = pd.read_sql("SELECT * FROM flats", local_conn)
print(f"Loaded {len(df_sales)} flats locally.")
df_rentals = pd.read_sql("SELECT * FROM rentals", local_conn)
print(f"Loaded {len(df_rentals)} rentals locally.")
local_conn.close()

print("\n2. Connecting to Supabase Pooler...")
supa_uri = "postgresql://postgres.ipmcnhuncxsybbtzgbfd:myCAqEgQMpiGVzB1@aws-1-eu-central-1.pooler.supabase.com:5432/postgres?sslmode=require"
engine = create_engine(supa_uri)

print("Pushing 'flats' table...")
df_sales.to_sql('flats', getattr(engine, 'connect', lambda: engine)(), if_exists='replace', index=False, method='multi', chunksize=500)

print("Pushing 'rentals' table...")
df_rentals.to_sql('rentals', getattr(engine, 'connect', lambda: engine)(), if_exists='replace', index=False, method='multi', chunksize=500)

print("\n3. Verification Query on Supabase:")
with engine.connect() as conn:
    check_sales = pd.read_sql("SELECT COUNT(*) FROM flats", conn)
    check_rentals = pd.read_sql("SELECT COUNT(*) FROM rentals", conn)
    print(f"Flats verified on cloud: {check_sales.iloc[0,0]} rows")
    print(f"Rentals verified on cloud: {check_rentals.iloc[0,0]} rows")
