import os
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
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
    price, title, usable_area, disposition, district, floor, 
    building_type, building_condition, ownership_type, 
    lift, garage, balcony, terrace, loggia, address_full,
    latitude, longitude
FROM flats
WHERE price IS NOT NULL 
  AND price > 100000        -- basic filter for realistic prices
  AND price < 150000000     -- exclude ultra-luxury outliers
"""
df = pd.read_sql(query, conn)
conn.close()

# Attempt to extract missing usable_area from title (e.g. "Prodej bytu 1+kk 26 m²" -> 26)
extracted_area = df['title'].str.extract(r'(\d+)\s*m', expand=False).astype(float)
df['usable_area'] = df['usable_area'].combine_first(extracted_area)

# Filter out rows that are still missing usable_area or are too small
df = df[df['usable_area'].notnull() & (df['usable_area'] > 10)].copy()

# Consolidate outdoor amenities
df['has_outdoor'] = df['balcony'].fillna(False) | df['terrace'].fillna(False) | df['loggia'].fillna(False)

# Extract robust disposition from the title string (since API serves '0')
df['disposition'] = df['title'].str.extract(r'(\d\+(?:kk|1)|Atypický)', expand=False).fillna('Unknown')

# Clean district and ownership
df['district'] = df['district'].fillna('Unknown')
df['ownership_type'] = df['ownership_type'].fillna('Unknown')

# Gracefully impute coordinates if scrape is currently ongoing
df['latitude'] = df['latitude'].fillna(df['latitude'].median() if not df['latitude'].isna().all() else 50.088)
df['longitude'] = df['longitude'].fillna(df['longitude'].median() if not df['longitude'].isna().all() else 14.420)

print(f"Data loaded. Total valid rows: {len(df)}")

import matplotlib.pyplot as plt
import seaborn as sns

print("Generating EDA plots...")
os.makedirs("eda_plots", exist_ok=True)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='usable_area', y='price', alpha=0.5)
plt.title('Price vs Usable Area')
plt.savefig('eda_plots/price_vs_area.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='disposition', y='price')
plt.title('Price by Disposition')
plt.xticks(rotation=45)
plt.savefig('eda_plots/price_vs_disposition.png')
plt.close()

top_districts = df['district'].value_counts().nlargest(10).index
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df['district'].isin(top_districts)], x='district', y='price')
plt.title('Price by Top 10 Districts')
plt.xticks(rotation=45)
plt.savefig('eda_plots/price_vs_district.png')
plt.close()
print("EDA plots magically saved into local 'eda_plots/' directory.")

# ──────────────────────────────────────────────
# 2. Preprocessing Setup
# ──────────────────────────────────────────────
num_features = ['usable_area', 'floor', 'latitude', 'longitude']
cat_features = ['disposition', 'district', 'building_type', 'building_condition', 'ownership_type']
bool_features = ['lift', 'garage', 'has_outdoor']

# Fill boolean NAs with False
for col in bool_features:
    df[col] = df[col].fillna(False).astype(int)

X = df[num_features + cat_features + bool_features]
# Log target because price is heavily skewed
y = np.log1p(df['price'])

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
def print_metrics(model_name, y_true_czk, y_pred_czk, save=False):
    rmse = np.sqrt(mean_squared_error(y_true_czk, y_pred_czk))
    mae = mean_absolute_error(y_true_czk, y_pred_czk)
    mape = mean_absolute_percentage_error(y_true_czk, y_pred_czk)
    r2 = r2_score(y_true_czk, y_pred_czk)
    print(f"--- {model_name} ---")
    print(f"RMSE: {rmse:,.0f} CZK | MAE: {mae:,.0f} CZK | MAPE: {mape:.2%} | R2: {r2:.4f}\n")
    
    if save:
        metrics_dict = {"rmse": rmse, "mae": mae, "mape": float(mape), "r2": float(r2)}
        with open('metrics.json', 'w') as f:
            json.dump(metrics_dict, f)
            
    return mape

y_test_czk = np.expm1(y_test)

# A. Baseline: Linear Regression
lr_pipeline = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
lr_pipeline.fit(X_train, y_train)
lr_pred_czk = np.expm1(lr_pipeline.predict(X_test))
print_metrics("Linear Regression Baseline", y_test_czk, lr_pred_czk)

# B. XGBoost Regressor (with RandomizedSearchCV Hyperparameter Tuning optimizing MAPE)
print("Tuning XGBoost...")
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor), 
    ('model', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))
])

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7, 9],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}

# We use negative mean absolute error for tuning
search = RandomizedSearchCV(
    xgb_pipeline, 
    param_distributions=param_grid, 
    n_iter=10, 
    scoring='neg_mean_absolute_percentage_error', 
    cv=3, 
    verbose=1, 
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
best_pred_czk = np.expm1(best_model.predict(X_test))
print(f"Best XGBoost Params: {search.best_params_}")
print_metrics("Tuned XGBoost Regressor", y_test_czk, best_pred_czk, save=True)

# Save the absolute best model
joblib.dump(best_model, "sreality_price_model.pkl")
print("\nFinal Tuned Model saved locally to 'sreality_price_model.pkl'.")
