"""
Train a LightGBM rental price prediction model.
================================================
Queries the 'rentals' table, engineers features, tunes a LightGBM regressor
using RandomizedSearchCV, and saves the model + metrics.

Usage
-----
    python train_model_rentals.py
"""
import os
import re
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
import joblib
import json

# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
load_dotenv()
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432'),
    dbname=os.getenv('DB_NAME', 'sreality'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', '')
)

query = """
SELECT 
    monthly_rent, title, usable_area, disposition, district, floor, 
    building_type, building_condition, furnished, location_type,
    lift, garage, balcony, terrace, loggia, address_full,
    latitude, longitude, energy_class
FROM rentals
WHERE monthly_rent IS NOT NULL 
  AND monthly_rent > 3000        -- Filter out unrealistic low prices
  AND monthly_rent <= 100000     -- Filter ultra-luxury outliers
"""
df = pd.read_sql(query, conn)
conn.close()

print(f"Loaded {len(df)} rental listings from database.")

if len(df) < 20:
    print("ERROR: Not enough data to train a model. Run the scraper first!")
    print("  python scraper_rentals.py --pages 0")
    exit(1)

# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

# Extract usable_area from title if missing (e.g. "Pronájem bytu 2+kk 55 m²" → 55)
extracted_area = df['title'].str.extract(r'(\d+)\s*m', expand=False).astype(float)
df['usable_area'] = df['usable_area'].combine_first(extracted_area)

# Filter out rows that are still missing usable_area, too small, or hyper-large
df = df[df['usable_area'].notnull() & (df['usable_area'] > 10) & (df['usable_area'] <= 300)].copy()

# Consolidate outdoor amenities
df['has_outdoor'] = df['balcony'].fillna(False) | df['terrace'].fillna(False) | df['loggia'].fillna(False)

# Extract disposition from title since API often returns 0 for sub_cb
title_disp = df['title'].str.extract(r'(\d\+(?:kk|1)|Atypický)', expand=False)
df['disposition'] = df['disposition'].where(
    df['disposition'].notna() & (df['disposition'] != '0') & (df['disposition'] != 'None'),
    title_disp
).fillna('Unknown')

# Clean categorical columns
df['district'] = df['district'].fillna('Unknown')
df['building_type'] = df['building_type'].fillna('Unknown')
df['building_condition'] = df['building_condition'].fillna('Unknown')
df['furnished'] = df['furnished'].fillna('Unknown')
df['location_type'] = df['location_type'].fillna('Unknown')

# Clean energy class
df['energy_class'] = df['energy_class'].astype(str).str.extract(r'(?:Třída )?([A-G])', expand=False).fillna('Unknown')
energy_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'Unknown': 4}
df['energy_num'] = df['energy_class'].map(energy_map)

# Impute coordinates
df['latitude'] = df['latitude'].fillna(df['latitude'].median() if not df['latitude'].isna().all() else 50.088)
df['longitude'] = df['longitude'].fillna(df['longitude'].median() if not df['longitude'].isna().all() else 14.420)

# Neighborhood extraction from address
df['neighborhood'] = df['address_full'].astype(str).apply(lambda x: x.split('-')[-1].strip() if '-' in x else 'Unknown')
df['neighborhood'] = df['neighborhood'].replace(['None', 'none', 'null', 'nan', ''], 'Unknown')

# Haversine distance to Prague Old Town Center (50.087, 14.421)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

df['distance_to_center_km'] = haversine(df['latitude'], df['longitude'], 50.087, 14.421)

print(f"Data after cleaning. Total valid rows: {len(df)}")

# ──────────────────────────────────────────────
# EDA Plots
# ──────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("Generating rental EDA plots...")
os.makedirs("eda_plots", exist_ok=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='usable_area', y='monthly_rent', alpha=0.5)
plt.title('Monthly Rent vs Usable Area')
plt.ylabel('Monthly Rent (CZK)')
plt.savefig('eda_plots/rental_rent_vs_area.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='disposition', y='monthly_rent')
plt.title('Monthly Rent by Disposition')
plt.ylabel('Monthly Rent (CZK)')
plt.xticks(rotation=45)
plt.savefig('eda_plots/rental_rent_vs_disposition.png')
plt.close()

if df['furnished'].nunique() > 1:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='furnished', y='monthly_rent')
    plt.title('Monthly Rent by Furnishing Level')
    plt.ylabel('Monthly Rent (CZK)')
    plt.xticks(rotation=45)
    plt.savefig('eda_plots/rental_rent_vs_furnished.png')
    plt.close()

print("EDA plots saved to 'eda_plots/' directory.")

# ──────────────────────────────────────────────
# 2. Preprocessing Setup
# ──────────────────────────────────────────────
num_features = ['usable_area', 'floor', 'latitude', 'longitude', 'distance_to_center_km', 'energy_num']
cat_features = ['disposition', 'district', 'neighborhood', 'building_type', 'building_condition', 'furnished', 'location_type']
bool_features = ['lift', 'garage', 'has_outdoor']

# Fill boolean NAs with False
for col in bool_features:
    df[col] = df[col].fillna(False).astype(int)

X = df[num_features + cat_features + bool_features]
# Log target because rent is skewed
y = np.log1p(df['monthly_rent'])

# Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformers
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features),
    ('bool', 'passthrough', bool_features)
])

# ──────────────────────────────────────────────
# 3. Model Training & Evaluation
# ──────────────────────────────────────────────
def print_metrics(model_name, y_true_czk, y_pred_czk, save_path=None):
    rmse = np.sqrt(mean_squared_error(y_true_czk, y_pred_czk))
    mae = mean_absolute_error(y_true_czk, y_pred_czk)
    mape = mean_absolute_percentage_error(y_true_czk, y_pred_czk)
    r2 = r2_score(y_true_czk, y_pred_czk)
    print(f"--- {model_name} ---")
    print(f"RMSE: {rmse:,.0f} CZK | MAE: {mae:,.0f} CZK | MAPE: {mape:.2%} | R2: {r2:.4f}\n")
    
    if save_path:
        metrics_dict = {"rmse": rmse, "mae": mae, "mape": float(mape), "r2": float(r2)}
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f)
            
    return mape

y_test_czk = np.expm1(y_test)

# A. Baseline: Linear Regression
print("Training Linear Regression baseline...")
lr_pipeline = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
lr_pipeline.fit(X_train, y_train)
lr_pred_czk = np.expm1(lr_pipeline.predict(X_test))
print_metrics("Linear Regression Baseline", y_test_czk, lr_pred_czk)

# B. LightGBM Regressor (with RandomizedSearchCV)
print("Tuning LightGBM for rental pricing...")
lgb_pipeline = Pipeline([
    ('preprocessor', preprocessor), 
    ('model', lgb.LGBMRegressor(random_state=42, verbose=-1, objective='regression_l1'))
])

param_grid = {
    'model__n_estimators': [100, 200, 300, 500],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7, 9],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0],
    'model__num_leaves': [31, 50, 70]
}

search = RandomizedSearchCV(
    lgb_pipeline, 
    param_distributions=param_grid, 
    n_iter=10, 
    scoring='neg_mean_absolute_error', 
    cv=3, 
    verbose=1, 
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
best_pred_czk = np.expm1(best_model.predict(X_test))
print(f"Best LightGBM Params: {search.best_params_}")
print_metrics("Tuned LightGBM Regressor (Rentals)", y_test_czk, best_pred_czk, save_path='rental_metrics.json')

# Save model
joblib.dump(best_model, "sreality_rental_model.pkl")
print("\nRental model saved to 'sreality_rental_model.pkl'.")

# ──────────────────────────────────────────────
# 4. Feature Importance Diagnostics
# ──────────────────────────────────────────────
print("Generating Feature Importance diagnostics...")
cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
encoded_cats = list(cat_encoder.get_feature_names_out(cat_features))
all_features = num_features + encoded_cats + bool_features

importances = best_model.named_steps['model'].feature_importances_
fi_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(data=fi_df, x='Importance', y='Feature', palette='magma')
plt.title('Top 20 Driving Predictive Features — Rental Model (LightGBM)')
plt.tight_layout()
plt.savefig('eda_plots/rental_feature_importance.png')
plt.close()
print("Rental feature importance saved to 'eda_plots/rental_feature_importance.png'!")
