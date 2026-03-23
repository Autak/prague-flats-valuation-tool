import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import os
import joblib
import json
from dotenv import load_dotenv

# Page config
st.set_page_config(page_title="Sreality Pricing Dashboard", layout="wide", page_icon="🏢")

# Title
st.title("🇨🇿 Prague Flats Price Predictor & Dashboard")

# Connect to DB
@st.cache_resource
def get_db_connection():
    load_dotenv()
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'sreality'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '')
    )
    return conn

# Load model
@st.cache_resource
def load_model():
    if os.path.exists("sreality_price_model.pkl"):
        return joblib.load("sreality_price_model.pkl")
    return None

model = load_model()
conn = get_db_connection()

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Database Explorer", "🤖 Model Metrics", "🔮 Price Predictor"])

# ──────────────────────────────────────────────
# Tab 1: Database Explorer
# ──────────────────────────────────────────────
with tab1:
    st.header("Latest Scraped Flats")
    
    # Load a sample from DB
    @st.cache_data(ttl=600)
    def fetch_data():
        query = """
        SELECT hash_id, title, price, usable_area, disposition, district, building_type, floor, building_condition, 
               ownership_type, lift, garage, balcony, terrace, loggia, address_full, scraped_at, listing_url
        FROM flats
        WHERE price IS NOT NULL AND title IS NOT NULL
        ORDER BY updated_at DESC
        LIMIT 1000
        """
        df = pd.read_sql(query, conn)
        
        # Consolidate outdoor amenities for easier viewing
        df['outdoor'] = df['balcony'].fillna(False) | df['terrace'].fillna(False) | df['loggia'].fillna(False)
        df.drop(columns=['balcony', 'terrace', 'loggia'], inplace=True)
        
        # Reformat boolean logic to be readable
        df['lift'] = df['lift'].fillna(False)
        df['garage'] = df['garage'].fillna(False)
        
        # Extract correct disposition from title because the API failed to fetch it (recorded as 0)
        df['disposition'] = df['title'].str.extract(r'(\d\+(?:kk|1)|Atypický)', expand=False).fillna('Unknown')
        # Fix missing usable_area from title
        missing = df['usable_area'].isnull()
        if missing.any():
            extracted = df.loc[missing, 'title'].str.extract(r'(\d+)\s*m', expand=False).astype(float)
            df.loc[missing, 'usable_area'] = extracted
        return df
    
    data = fetch_data()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows Fetched", len(data))
    if not data.empty:
        col2.metric("Median Price", f"{data['price'].median():,.0f} CZK")
        col3.metric("Median Area", f"{data['usable_area'].median():,.0f} m²")
    
    st.dataframe(data, use_container_width=True, hide_index=True)
    
    st.subheader("Prague Flat Geolocations")
    @st.cache_data(ttl=600)
    def fetch_map_data():
        query = "SELECT latitude as lat, longitude as lon FROM flats WHERE latitude IS NOT NULL"
        return pd.read_sql(query, conn)
        
    map_data = fetch_map_data()
    if not map_data.empty:
        st.map(map_data)
    else:
        st.info("Map data will dynamically populate as the background scraper saves exact coordinates!")

# ──────────────────────────────────────────────
# Tab 2: Model Metrics
# ──────────────────────────────────────────────
with tab2:
    st.header("Machine Learning Performance")
    st.markdown("""
    Based on the latest hyperparameter tuning run, an **XGBoost Regressor** is dynamically trained to minimize absolute relative errors (MAPE).
    
    The model translates the flat's raw attributes (Usable Area, Floor, Neighborhood, Property Type) to output the predicted listing price. 
    """)

    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{metrics['rmse']:,.0f} CZK")
        c2.metric("Mean Ablst. Error (MAE)", f"{metrics['mae']:,.0f} CZK")
        c3.metric("MAPE (Relative Error)", f"{metrics['mape']:.2%}")
        c4.metric("R² Correlation", f"{metrics['r2']:.4f}")
    else:
        st.info("Continuous accuracy metrics will populate here automatically once `train_model.py` finishes executing!")

# ──────────────────────────────────────────────
# Tab 3: Price Predictor
# ──────────────────────────────────────────────
with tab3:
    st.header("Predict Your Flat's Market Listing Price")
    
    if model is None:
        st.error("Model file not found! Please run the training script first.")
    else:
        with st.form("prediction_form"):
            st.subheader("Flat Details")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                area = st.number_input("Usable Area (m²)", min_value=10.0, max_value=500.0, value=50.0)
                floor = st.number_input("Floor", min_value=-2, max_value=30, value=2)
                disposition = st.selectbox("Disposition", ["1+kk", "1+1", "2+kk", "2+1", "3+kk", "3+1", "4+kk", "4+1", "5+kk", "Atypický"])
            
            with c2:
                district = st.selectbox("District", [f"Praha {i}" for i in range(1, 11)] + ["Praha-východ", "Praha-západ", "Other"])
                st.markdown("**Exact Coordinates**")
                lat_input = st.number_input("Latitude", value=50.088, format="%.5f")
                lon_input = st.number_input("Longitude", value=14.420, format="%.5f")
            
            with c3:
                building_type = st.selectbox("Building Type", ["Cihlová", "Panelová", "Smíšená", "Skeletová", "Novostavba"])
                building_condition = st.selectbox("Condition", ["Velmi dobrý", "Dobrý", "Po rekonstrukci", "Před rekonstrukcí", "Novostavba", "Projekt"])
                ownership = st.selectbox("Ownership", ["Osobní", "Družstevní", "Státní/obecní"])
                st.markdown("**Amenities**")
                has_lift = st.checkbox("Has Lift", value=True)
                has_garage = st.checkbox("Has Garage/Parking")
                has_outdoor = st.checkbox("Has Balcony/Terrace/Loggia")
            
            submitted = st.form_submit_button("Predict Price!", type="primary", use_container_width=True)
            
            if submitted:
                # Build dataframe matching training schema
                input_df = pd.DataFrame([{
                    "usable_area": area,
                    "floor": floor,
                    "latitude": lat_input,
                    "longitude": lon_input,
                    "disposition": disposition,
                    "district": district,
                    "building_type": building_type,
                    "building_condition": building_condition,
                    "ownership_type": ownership,
                    "lift": 1 if has_lift else 0,
                    "garage": 1 if has_garage else 0,
                    "has_outdoor": 1 if has_outdoor else 0
                }])
                
                # Predict (returns log1p of price)
                pred_log = model.predict(input_df)[0]
                pred_czk = np.expm1(pred_log)
                
                st.success("Prediction Complete!")
                st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{pred_czk:,.0f} CZK</h1>", unsafe_allow_html=True)
                
                # Categorize
                if pred_czk < 5_000_000:
                    bucket = "< 5M CZK (Budget)"
                elif pred_czk < 8_000_000:
                    bucket = "5M - 8M CZK (Mid-range)"
                elif pred_czk < 12_000_000:
                    bucket = "8M - 12M CZK (Premium)"
                else:
                    bucket = "> 12M CZK (Luxury)"
                    
                st.markdown(f"<h4 style='text-align: center;'>Market Bracket: {bucket}</h4>", unsafe_allow_html=True)
