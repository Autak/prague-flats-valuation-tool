import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import os
import joblib
import json
import pydeck as pdk
import plotly.express as px
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sreality Pricing Dashboard", layout="wide", page_icon="🏢")

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

def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

sales_model = load_model("sreality_price_model.pkl")
rental_model = load_model("sreality_rental_model.pkl")
conn = get_db_connection()

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def haversine(lat1, lon1, lat2, lon2):
    return haversine_vectorized(lat1, lon1, lat2, lon2)

def button_group(label, options, key, sidebar=False):
    container = st.sidebar if sidebar else st
    
    if key not in st.session_state:
        st.session_state[key] = options[0]
        
    if label:
        container.markdown(f"<p style='font-size: 14px; font-weight: 600; margin-bottom: 5px;'>{label}</p>", unsafe_allow_html=True)
        
    cols = container.columns(len(options))
    for i, opt in enumerate(options):
        b_type = "primary" if st.session_state[key] == opt else "secondary"
        if cols[i].button(opt, key=f"{key}_{i}", use_container_width=True, type=b_type):
            st.session_state[key] = opt
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
                
    container.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    return st.session_state[key]

# ─────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_sales_data():
    query = """
    SELECT hash_id, title, price, usable_area, disposition, district, building_type, floor, building_condition, 
           ownership_type, energy_class, lift, garage, balcony, terrace, loggia, address_full, latitude, longitude
    FROM flats
    WHERE price IS NOT NULL AND title IS NOT NULL AND latitude IS NOT NULL
      AND price <= 45000000
    ORDER BY updated_at DESC
    """
    df = pd.read_sql(query, conn)
    
    df['disposition'] = df['title'].str.extract(r'(\d\+(?:kk|1)|Atypický)', expand=False).fillna('Unknown')
    df['energy_class'] = df['energy_class'].astype(str).str.extract(r'(?:Třída )?([A-G])', expand=False).fillna('Unknown')
    
    missing_area = df['usable_area'].isnull()
    if missing_area.any():
        df.loc[missing_area, 'usable_area'] = df.loc[missing_area, 'title'].str.extract(r'(\d+)\s*m', expand=False).astype(float)
    df = df[df['usable_area'].notnull() & (df['usable_area'] > 10)].copy()
    
    df['neighborhood'] = df['address_full'].astype(str).apply(lambda x: x.split('-')[-1].strip() if '-' in x else 'Unknown')
    energy_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'Unknown': 4}
    df['energy_num'] = df['energy_class'].map(energy_map).fillna(4)
    
    df['lift'] = df['lift'].fillna(False).astype(int)
    df['garage'] = df['garage'].fillna(False).astype(int)
    df['has_outdoor'] = (df['balcony'].fillna(False) | df['terrace'].fillna(False) | df['loggia'].fillna(False)).astype(int)
    
    df['distance_to_center_km'] = haversine_vectorized(df['latitude'], df['longitude'], 50.087, 14.421)
    df['price_per_sqm'] = df['price'] / df['usable_area']
    
    if sales_model is not None:
        features = df[['usable_area', 'floor', 'latitude', 'longitude', 'distance_to_center_km', 'energy_num',
                       'disposition', 'district', 'neighborhood', 'building_type', 'building_condition',
                       'ownership_type', 'lift', 'garage', 'has_outdoor']].copy()
        
        features['floor'] = pd.to_numeric(features['floor'], errors='coerce').fillna(2)
        features['building_type'] = features['building_type'].fillna('Unknown')
        features['building_condition'] = features['building_condition'].fillna('Unknown')
        features['ownership_type'] = features['ownership_type'].fillna('Unknown')
        features['district'] = features['district'].fillna('Unknown')
        
        preds_log = sales_model.predict(features)
        df['predicted_price'] = np.expm1(preds_log)
        df['market_delta_raw'] = (df['price'] - df['predicted_price']) / df['predicted_price']
        
        def format_delta(val):
            if pd.isna(val): return None
            s = f"{val:+.0%}"
            return f"{s} [green]" if val < 0 else f"{s} [red]"
            
        df['market_delta'] = df['market_delta_raw'].apply(format_delta)
    else:
        df['predicted_price'] = np.nan
        df['market_delta'] = None
        df['market_delta_raw'] = np.nan
        
    return df


@st.cache_data(ttl=300)
def load_rentals_data():
    # Check if rentals table exists
    try:
        test_q = "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'rentals'"
        test_df = pd.read_sql(test_q, conn)
        if test_df.iloc[0, 0] == 0:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    query = """
    SELECT hash_id, title, monthly_rent, usable_area, disposition, district, building_type, floor, 
           building_condition, furnished, location_type, energy_class,
           lift, garage, balcony, terrace, loggia, address_full, latitude, longitude
    FROM rentals
    WHERE monthly_rent IS NOT NULL AND title IS NOT NULL AND latitude IS NOT NULL
      AND monthly_rent <= 100000
    ORDER BY updated_at DESC
    """
    df = pd.read_sql(query, conn)
    
    if df.empty:
        return df
    
    # Extract disposition from title
    title_disp = df['title'].str.extract(r'(\d\+(?:kk|1)|Atypický)', expand=False)
    df['disposition'] = df['disposition'].where(
        df['disposition'].notna() & (df['disposition'] != '0') & (df['disposition'] != 'None'),
        title_disp
    ).fillna('Unknown')
    
    df['energy_class'] = df['energy_class'].astype(str).str.extract(r'(?:Třída )?([A-G])', expand=False).fillna('Unknown')
    
    missing_area = df['usable_area'].isnull()
    if missing_area.any():
        df.loc[missing_area, 'usable_area'] = df.loc[missing_area, 'title'].str.extract(r'(\d+)\s*m', expand=False).astype(float)
    df = df[df['usable_area'].notnull() & (df['usable_area'] > 10)].copy()
    
    df['neighborhood'] = df['address_full'].astype(str).apply(lambda x: x.split('-')[-1].strip() if '-' in x else 'Unknown')
    energy_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'Unknown': 4}
    df['energy_num'] = df['energy_class'].map(energy_map).fillna(4)
    
    df['lift'] = df['lift'].fillna(False).astype(int)
    df['garage'] = df['garage'].fillna(False).astype(int)
    df['has_outdoor'] = (df['balcony'].fillna(False) | df['terrace'].fillna(False) | df['loggia'].fillna(False)).astype(int)
    df['furnished'] = df['furnished'].fillna('Unknown')
    df['location_type'] = df['location_type'].fillna('Unknown')
    
    df['distance_to_center_km'] = haversine_vectorized(df['latitude'], df['longitude'], 50.087, 14.421)
    df['rent_per_sqm'] = df['monthly_rent'] / df['usable_area']
    
    if rental_model is not None:
        features = df[['usable_area', 'floor', 'latitude', 'longitude', 'distance_to_center_km', 'energy_num',
                       'disposition', 'district', 'neighborhood', 'building_type', 'building_condition',
                       'furnished', 'location_type', 'lift', 'garage', 'has_outdoor']].copy()
        
        features['floor'] = pd.to_numeric(features['floor'], errors='coerce').fillna(2)
        features['building_type'] = features['building_type'].fillna('Unknown')
        features['building_condition'] = features['building_condition'].fillna('Unknown')
        features['district'] = features['district'].fillna('Unknown')
        
        preds_log = rental_model.predict(features)
        df['predicted_rent'] = np.expm1(preds_log)
        df['market_delta_raw'] = (df['monthly_rent'] - df['predicted_rent']) / df['predicted_rent']
        
        def format_delta(val):
            if pd.isna(val): return None
            s = f"{val:+.0%}"
            return f"{s} [green]" if val < 0 else f"{s} [red]"
            
        df['market_delta'] = df['market_delta_raw'].apply(format_delta)
    else:
        df['predicted_rent'] = np.nan
        df['market_delta'] = None
        df['market_delta_raw'] = np.nan
        
    return df


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
sales_df = load_sales_data()
rentals_df = load_rentals_data()
has_rentals = not rentals_df.empty

# ─────────────────────────────────────────────────────────
# NAVIGATION SIDEBAR
# ─────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = button_group("Go to", ["Dashboard", "Prediction Engine"], "nav_page", sidebar=True)

# ═══════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ═══════════════════════════════════════════════════════════
if page == "Dashboard":
    # Market mode toggle
    st.sidebar.markdown("---")
    if has_rentals:
        market_mode = button_group("Market", ["Flats for Sale", "Flats for Rent"], "dash_market", sidebar=True)
    else:
        market_mode = "Flats for Sale"
        st.sidebar.info("Rental data not available. Run `scraper_rentals.py` first.")
    
    is_rental = market_mode == "Flats for Rent"
    raw_df = rentals_df if is_rental else sales_df
    price_col = 'monthly_rent' if is_rental else 'price'
    pred_col = 'predicted_rent' if is_rental else 'predicted_price'
    psqm_col = 'rent_per_sqm' if is_rental else 'price_per_sqm'
    price_label = "Monthly Rent (CZK)" if is_rental else "Price (CZK)"
    
    # ── FILTERS ──
    st.sidebar.markdown("---")
    st.sidebar.header("Filters")
    
    if not raw_df.empty:
        min_price, max_price = st.sidebar.slider(
            price_label, 
            int(raw_df[price_col].min()), int(raw_df[price_col].max()), 
            (int(raw_df[price_col].min()), int(raw_df[price_col].max()))
        )
        min_area, max_area = st.sidebar.slider(
            "Usable Area (m²)", 
            int(raw_df['usable_area'].min()), int(raw_df['usable_area'].max()), 
            (int(raw_df['usable_area'].min()), int(raw_df['usable_area'].max()))
        )
        valid_districts = sorted([d for d in raw_df['district'].unique() if pd.notnull(d) and d != 'Unknown'])
        districts = st.sidebar.multiselect("District", options=valid_districts, default=valid_districts)
        
        conds_opt = sorted([c for c in raw_df['building_condition'].astype(str).unique() if c and c != 'nan'])
        conditions = st.sidebar.multiselect("Building Condition", options=conds_opt, default=conds_opt)
        
        energy_opt = sorted([e for e in raw_df['energy_class'].astype(str).unique() if e and e != 'nan'])
        energy = st.sidebar.multiselect("Energy Class", options=energy_opt, default=energy_opt)
        
        if is_rental:
            furn_opt = sorted([f for f in raw_df['furnished'].astype(str).unique() if f and f != 'nan' and f != 'Unknown'])
            if furn_opt:
                furnished_filter = st.sidebar.multiselect("Furnished", options=furn_opt, default=furn_opt)
        
        req_lift = st.sidebar.toggle("Lift required")
        req_garage = st.sidebar.toggle("Garage required")
        
        df = raw_df[
            (raw_df[price_col] >= min_price) & (raw_df[price_col] <= max_price) &
            (raw_df['usable_area'] >= min_area) & (raw_df['usable_area'] <= max_area) &
            (raw_df['district'].isin(districts) if districts else True) &
            (raw_df['building_condition'].astype(str).isin(conditions) if conditions else True) &
            (raw_df['energy_class'].astype(str).isin(energy) if energy else True)
        ]
        if is_rental and furn_opt:
            df = df[df['furnished'].astype(str).isin(furnished_filter)]
        if req_lift:
            df = df[df['lift'] == 1]
        if req_garage:
            df = df[df['garage'] == 1]
    else:
        df = raw_df

    # ── TOP TABLE ──
    header_text = "Rental Listings — Live Market Table" if is_rental else "Algorithmic Predictor Live Table"
    st.subheader(header_text)
    
    def color_delta_text(val):
        if pd.isna(val) or val == "":
            return ''
        color = '#4CAF50' if '[green]' in str(val) else '#F44336'
        return f'color: {color}; font-weight: bold;'
    
    if not df.empty:
        if is_rental:
            display_cols = ['hash_id', 'title', 'monthly_rent', 'usable_area', 'disposition', 'district', 
                           'building_type', 'floor', 'building_condition', 'furnished', pred_col, 'market_delta', 'lift']
            format_dict = {
                'monthly_rent': '{:,.0f}',
                'usable_area': '{:.1f} m²',
                pred_col: '{:,.0f} CZK/mo'
            }
        else:
            display_cols = ['hash_id', 'title', 'price', 'usable_area', 'disposition', 'district', 
                           'building_type', 'floor', 'building_condition', 'ownership_type', pred_col, 'market_delta', 'lift']
            format_dict = {
                'price': '{:,.0f}',
                'usable_area': '{:.1f} m²',
                pred_col: '{:,.0f} CZK'
            }
        
        # Only include columns that exist
        display_cols = [c for c in display_cols if c in df.columns]
        styled_df = df[display_cols].style.format(format_dict).map(color_delta_text, subset=['market_delta'])
        st.dataframe(styled_df, use_container_width=True, height=250, hide_index=True)
    else:
        st.info("No data available for the current filters.")

    # ── MAP ──
    map_title = "Prague Rental Geolocations" if is_rental else "Prague Flat Geolocations"
    st.subheader(map_title)
    if not df.empty and psqm_col in df.columns:
        min_psqm = df[psqm_col].min()
        max_psqm = df[psqm_col].quantile(0.90)
        df['norm_psqm'] = np.clip((df[psqm_col] - min_psqm) / (max_psqm - min_psqm + 1e-5), 0, 1)
        
        def get_rgb(v):
            if v < 0.5:
                return [int(v*2 * 255), 255, 0, 160]
            else:
                return [255, int((1.0-v)*2 * 255), 0, 160]
                
        df['color'] = df['norm_psqm'].apply(get_rgb)
        
        view_state = pdk.ViewState(latitude=50.087, longitude=14.421, zoom=10.5, pitch=0)
        
        map_layers = []
        if os.path.exists("prague_boundary.geojson"):
            import json
            with open("prague_boundary.geojson", "r", encoding="utf-8") as f:
                geo_data = json.load(f)
            border_layer = pdk.Layer(
                "GeoJsonLayer",
                data=geo_data,
                opacity=0.8,
                stroked=True,
                filled=False,
                get_line_color=[150, 150, 150, 255],
                line_width_min_pixels=2
            )
            map_layers.append(border_layer)
            
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[longitude, latitude]",
            get_radius=50,
            radius_min_pixels=3,
            radius_max_pixels=12,
            get_fill_color="color",
            pickable=True
        )
        map_layers.append(scatter_layer)
        
        if is_rental:
            tooltip = {"html": "<b>{title}</b><br/>Area: {usable_area} sqm<br/>Rent: <b>{monthly_rent} CZK/mo</b><br/>Predicted: <b>{predicted_rent} CZK/mo</b><br/>Furnished: {furnished}"}
        else:
            tooltip = {"html": "<b>{title}</b><br/>Area: {usable_area} sqm<br/>Current Price: <b>{price} CZK</b><br/>Predicted Avg: <b>{predicted_price} CZK</b><br/>Lifts: {lift}"}
        
        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            initial_view_state=view_state,
            layers=map_layers,
            tooltip=tooltip
        ))
    else:
        st.info("No matching flats found for the active filter set.")

    # ── BOTTOM INSIGHTS ──
    st.markdown("---")
    st.subheader("Market Insights")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Active Listings", f"{len(df):,}")
    k2.metric("Avg Area", f"{df['usable_area'].mean():.0f} sqm" if not df.empty else "0 sqm")
    
    if is_rental:
        k3.metric("Average Rent", f"{df['monthly_rent'].mean()/1e3:.1f}K CZK/mo" if not df.empty else "N/A")
        k4.metric("Predicted Average", f"{df['predicted_rent'].mean()/1e3:.1f}K CZK/mo" if not df.empty and 'predicted_rent' in df and df['predicted_rent'].notna().any() else "N/A")
    else:
        k3.metric("Average Price", f"{df['price'].mean()/1e6:.1f}M CZK" if not df.empty else "0M CZK")
        k4.metric("Predicted Average", f"{df['predicted_price'].mean()/1e6:.1f}M CZK" if not df.empty and 'predicted_price' in df and df['predicted_price'].notna().any() else "N/A")
    
    c_chart1, c_chart2 = st.columns([1.5, 1])
    with c_chart1:
        if is_rental:
            st.markdown("<p style='font-size: 14px; margin-bottom: 0px;'>Average Asking Rent vs. Predicted Rent by District</p>", unsafe_allow_html=True)
            if not df.empty and 'predicted_rent' in df:
                dist_grp = df.groupby('district')[['monthly_rent', 'predicted_rent']].mean().reset_index()
                dist_grp['rent_K'] = dist_grp['monthly_rent'] / 1e3
                dist_grp['pred_K'] = dist_grp['predicted_rent'] / 1e3
                
                fig1 = px.bar(
                    dist_grp, x='district', y=['rent_K', 'pred_K'], barmode='group',
                    color_discrete_map={'rent_K': '#FFA726', 'pred_K': '#42A5F5'}
                )
                fig1.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=250, 
                                  legend=dict(title="", orientation="h", y=1.2, x=0), 
                                  yaxis_title="Thousand CZK/mo", xaxis_title="")
                st.plotly_chart(fig1, use_container_width=True)
        else:
            st.markdown("<p style='font-size: 14px; margin-bottom: 0px;'>Average Asking Price vs. Predicted Price by District</p>", unsafe_allow_html=True)
            if not df.empty and 'predicted_price' in df:
                dist_grp = df.groupby('district')[['price', 'predicted_price']].mean().reset_index()
                dist_grp['price_M'] = dist_grp['price'] / 1e6
                dist_grp['pred_M'] = dist_grp['predicted_price'] / 1e6
                
                fig1 = px.bar(
                    dist_grp, x='district', y=['price_M', 'pred_M'], barmode='group',
                    color_discrete_map={'price_M': '#FFA726', 'pred_M': '#42A5F5'}
                )
                fig1.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=250, 
                                  legend=dict(title="", orientation="h", y=1.2, x=0), 
                                  yaxis_title="Million CZK", xaxis_title="")
                st.plotly_chart(fig1, use_container_width=True)
            
    with c_chart2:
        if is_rental:
            st.markdown("<p style='font-size: 14px; margin-bottom: 0px;'>Breakdown by Furnishing Level</p>", unsafe_allow_html=True)
            if not df.empty:
                furn_counts = df['furnished'].value_counts().reset_index()
                furn_counts.columns = ['Furnished', 'Count']
                fig2 = px.pie(furn_counts, names='Furnished', values='Count', hole=0.0)
                fig2.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=250, legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown("<p style='font-size: 14px; margin-bottom: 0px;'>Breakdown of Building Conditions</p>", unsafe_allow_html=True)
            if not df.empty:
                cond_counts = df['building_condition'].value_counts().reset_index()
                cond_counts.columns = ['Condition', 'Count']
                fig2 = px.pie(cond_counts, names='Condition', values='Count', hole=0.0)
                fig2.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=250, legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PREDICTION ENGINE PAGE
# ═══════════════════════════════════════════════════════════
elif page == "Prediction Engine":
    st.header("Advanced Prediction Engine")
    
    # Market mode toggle
    if has_rentals:
        pred_mode = button_group("Predict for:", ["Valuation of your flat", "Estimation of Rent"], "pred_market", sidebar=False)
    else:
        pred_mode = "Valuation of your flat"
    
    is_rental_pred = pred_mode == "Estimation of Rent"
    active_model = rental_model if is_rental_pred else sales_model
    active_df = rentals_df if is_rental_pred else sales_df
    
    if is_rental_pred:
        st.markdown("Input any known parameters of a rental flat to dynamically predict the monthly rent. **Note:** All string variables are entirely optional.")
    else:
        st.markdown("Input any known parameters of a flat to dynamically predict an accurate market valuation. **Note:** All string variables are entirely optional; the algorithm natively processes missing permutations dynamically.")
    
    with st.container(border=True):
        st.subheader("Property Characteristics")
        c1, c2, c3 = st.columns(3)
        valid_districts = sorted([d for d in active_df['district'].unique() if pd.notnull(d) and d != 'Unknown']) if not active_df.empty else []
        valid_neighborhoods = sorted([n for n in active_df['neighborhood'].unique() if pd.notnull(n) and n != 'Unknown']) if not active_df.empty else []
        
        with c1:
            if is_rental_pred:
                p_area = st.number_input("Usable Area (m²)", value=50, step=1)
            else:
                p_area = st.number_input("Usable Area (m²)", value=40, step=1)
            p_disp = st.selectbox("Disposition (Optional)", ["Unknown", "1+kk", "1+1", "2+kk", "2+1", "3+kk", "3+1", "4+kk"])
            p_dist = st.selectbox("District (Optional)", ["Unknown"] + valid_districts)
            p_neigh = st.selectbox("Neighborhood (Optional)", ["Unknown"] + valid_neighborhoods)
        with c2:
            p_floor = st.number_input("Floor Level", value=2, step=1)
            p_type = st.selectbox("Building Type (Optional)", ["Unknown", "Cihlová", "Panelová", "Smíšená", "Skeletová", "Novostavba"])
            p_cond = st.selectbox("Building Condition (Optional)", ["Unknown", "Velmi dobrý", "Dobrý", "Po rekonstrukci", "Před rekonstrukcí", "Novostavba", "Projekt"])
        with c3:
            p_energy = st.selectbox("Energy Class (Optional)", ["Unknown", "A", "B", "C", "D", "E", "F", "G"])
            
            if is_rental_pred:
                p_furn = st.selectbox("Furnished (Optional)", ["Unknown", "Vybavený", "Částečně vybavený", "Nevybavený"])
                p_loc_type = st.selectbox("Location Type (Optional)", ["Unknown", "Centrum obce", "Klidná část obce", "Okraj obce"])
            else:
                p_owner = st.selectbox("Ownership Type (Optional)", ["Unknown", "Osobní", "Družstevní", "Státní/obecní"])
            
            st.markdown("**Amenities**")
            p_lift = st.checkbox("Lift Access", value=True)
            p_garage = st.checkbox("Garage / Covered Parking")
            p_outdoor = st.checkbox("Balcony / Terrace / Loggia")
            
        if st.button("Predict Target Valuation", type="primary", use_container_width=True):
            if active_model is not None:
                # Lat/Lon resolving
                if p_neigh != "Unknown" and not active_df.empty:
                    neigh_sub = active_df[active_df['neighborhood'] == p_neigh]
                    if not neigh_sub.empty:
                        p_lat, p_lon = neigh_sub['latitude'].mean(), neigh_sub['longitude'].mean()
                    else:
                        p_lat, p_lon = 50.088, 14.420
                elif p_dist != "Unknown" and not active_df.empty:
                    dist_sub = active_df[active_df['district'] == p_dist]
                    if not dist_sub.empty:
                        p_lat, p_lon = dist_sub['latitude'].mean(), dist_sub['longitude'].mean()
                    else:
                        p_lat, p_lon = 50.088, 14.420
                else:
                    p_lat, p_lon = 50.088, 14.420
                    
                energy_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'Unknown': 4}
                
                if is_rental_pred:
                    in_df = pd.DataFrame([{
                        "usable_area": p_area,
                        "floor": p_floor,
                        "latitude": p_lat,
                        "longitude": p_lon,
                        "distance_to_center_km": haversine(p_lat, p_lon, 50.087, 14.421),
                        "energy_num": energy_map.get(p_energy, 4),
                        "disposition": p_disp,
                        "district": p_dist,
                        "neighborhood": p_neigh,
                        "building_type": p_type,
                        "building_condition": p_cond,
                        "furnished": p_furn,
                        "location_type": p_loc_type,
                        "lift": 1 if p_lift else 0,
                        "garage": 1 if p_garage else 0,
                        "has_outdoor": 1 if p_outdoor else 0
                    }])
                else:
                    in_df = pd.DataFrame([{
                        "usable_area": p_area,
                        "floor": p_floor,
                        "latitude": p_lat,
                        "longitude": p_lon,
                        "distance_to_center_km": haversine(p_lat, p_lon, 50.087, 14.421),
                        "energy_num": energy_map.get(p_energy, 4),
                        "disposition": p_disp,
                        "district": p_dist,
                        "neighborhood": p_neigh,
                        "building_type": p_type,
                        "building_condition": p_cond,
                        "ownership_type": p_owner,
                        "lift": 1 if p_lift else 0,
                        "garage": 1 if p_garage else 0,
                        "has_outdoor": 1 if p_outdoor else 0
                    }])
                
                pred_val = np.expm1(active_model.predict(in_df)[0])
                
                # Scale confidence metric
                conf = 95 - abs(p_area - 60)/10
                if p_dist == "Unknown": conf -= 15
                if p_disp == "Unknown": conf -= 5
                if p_cond == "Unknown": conf -= 5
                if is_rental_pred and p_furn == "Unknown": conf -= 5
                conf = min(max(conf, 40), 99)
                
                st.markdown("<hr/>", unsafe_allow_html=True)
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    if is_rental_pred:
                        st.markdown(f"<p style='color: #888; font-size:14px; margin-bottom: 0px;'>Estimated Monthly Rent</p>", unsafe_allow_html=True)
                        st.markdown(f"<h1 style='color: #4CAF50; margin-top: 5px;'>{pred_val:,.0f} CZK/mo</h1>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='color: #888; font-size:14px; margin-bottom: 0px;'>Estimated Market Value</p>", unsafe_allow_html=True)
                        st.markdown(f"<h1 style='color: #4CAF50; margin-top: 5px;'>{pred_val:,.0f} CZK</h1>", unsafe_allow_html=True)
                with res_col2:
                    st.markdown(f"<p style='color: #888; font-size:14px; margin-bottom: 0px;'>Prediction Confidence</p>", unsafe_allow_html=True)
                    score_color = "#4CAF50" if conf >= 80 else ("#FFA726" if conf >= 60 else "#F44336")
                    st.markdown(f"<h1 style='color: {score_color}; margin-top: 5px;'>{conf:.0f}%</h1>", unsafe_allow_html=True)
            else:
                model_name = "rental" if is_rental_pred else "sales"
                st.error(f"The {model_name} model file is not available! Run the training script first.")

    # ── ML METRICS & DIAGNOSTICS ──
    st.markdown("---")
    st.subheader("Algorithm Accuracy & Diagnostics")
    
    if is_rental_pred:
        metrics_path = "rental_metrics.json"
        importance_path = "eda_plots/rental_feature_importance.png"
        st.markdown("This LightGBM model predicts monthly rental prices based on property features, location, and furnishing level.")
    else:
        metrics_path = "metrics.json"
        importance_path = "eda_plots/feature_importance.png"
        st.markdown("This LightGBM model processes extremely strict validation bounds to securely predict your specific features. Here are its mathematically tested global performances across thousands of unseen properties.")
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        m1, m2, m3, m4 = st.columns(4)
        unit = "CZK/mo" if is_rental_pred else "CZK"
        m1.metric("Mean Absolute Error (MAE)", f"{metrics.get('mae', 0):,.0f} {unit}", help="Exact currency deviation across hidden test set")
        m2.metric("Relative Error (MAPE)", f"{metrics.get('mape', 0):.2%}", help="Average percentage mismatch between prediction and reality")
        m3.metric("Root Mean Squared Error", f"{metrics.get('rmse', 0):,.0f} {unit}")
        m4.metric("R² Correlation Math", f"{metrics.get('r2', 0):.4f}")
        
    c_diag1, c_diag2 = st.columns([1, 1])
    with c_diag1:
        if os.path.exists(importance_path):
            st.markdown("<br/><b>Driver Analytics (Feature Importance)</b>", unsafe_allow_html=True)
            if is_rental_pred:
                st.markdown("This chart shows which parameters the rental pricing model weighted most heavily when predicting monthly rents.", unsafe_allow_html=True)
            else:
                st.markdown("This chart perfectly displays exactly what parameters our Tree algorithm valued universally the most heavily when deciding prices.", unsafe_allow_html=True)
            st.image(importance_path, use_container_width=True)
