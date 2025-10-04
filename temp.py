import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot as plt
import pickle

# --- DATA PREPARATION (Replication of Step 1) ---

print("Starting model setup and saving process...")

# Load the Kepler Cumulative Data
file_name = "exoplanet-ai-data/cumulative_2025.10.03_22.54.46.csv"
df = pd.read_csv(file_name, skiprows=53)

TARGET = 'koi_disposition'

# 1. Define the full list of features to be used by the model (in desired order)
FULL_FEATURE_LIST = [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
    'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag', 'koi_score',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
]

# 2. Define the subset of features that REQUIRE IMPUTATION (continuous numerical data)
# The imputer MUST only be fit on columns that had NaNs, and the feature flags are NOT in this group.
BINARY_FLAG_COLS = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
IMPUTATION_FEATURE_LIST = [col for col in FULL_FEATURE_LIST if col not in BINARY_FLAG_COLS]

# Select all relevant columns from the raw data
df_features = df[[TARGET] + FULL_FEATURE_LIST].copy()
df_features.dropna(subset=[TARGET], inplace=True)

# 3. Apply Imputation ONLY to the defined subset of features
# Only these columns are passed to the imputer for fit/transform.
imputer = SimpleImputer(strategy='median')
df_features[IMPUTATION_FEATURE_LIST] = imputer.fit_transform(df_features[IMPUTATION_FEATURE_LIST])

# Check for remaining NaNs (should be none in the final feature set)
if df_features[FULL_FEATURE_LIST].isnull().sum().any():
    print("Warning: NaNs still present after imputation.")

# Target encoding
le = LabelEncoder()
df_features['disposition_encoded'] = le.fit_transform(df_features[TARGET])
class_names = le.classes_

# Final Feature/Target Definition
X = df_features[FULL_FEATURE_LIST] # Use the full, final list
y = df_features['disposition_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- MODEL TRAINING (Replication of Step 2) ---

print("Training Gradient Boosting Classifier...")
gbc_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gbc_model.fit(X_train, y_train)
print("Training complete.")

# --- SAVE ARTIFACTS ---

# 1. Save the trained Gradient Boosting Classifier model
with open('gbc_model.pkl', 'wb') as file:
    pickle.dump(gbc_model, file)
print("Saved 'gbc_model.pkl'")

# 2. Save the fitted SimpleImputer object and the list of columns it handles
# We save the Imputer and the list of columns it expects (IMPUTATION_FEATURE_LIST)
with open('imputer_cols.pkl', 'wb') as file:
    pickle.dump(IMPUTATION_FEATURE_LIST, file)

with open('imputer.pkl', 'wb') as file:
    pickle.dump(imputer, file)
print("Saved 'imputer.pkl' and 'imputer_cols.pkl'")


# 3. Save the full feature list in the correct order for the ML model
with open('feature_cols.pkl', 'wb') as file:
    pickle.dump(FULL_FEATURE_LIST, file)
print("Saved 'feature_cols.pkl'")

# 4. Save the Feature Importance Plot image
importance = gbc_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': FULL_FEATURE_LIST,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'][:15], feature_importance_df['Importance'][:15], color='darkred')
plt.xlabel('Feature Importance Score')
plt.title('Top 15 Most Important Features for Exoplanet Classification')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_plot.png')
print("Saved 'feature_importance_plot.png'")

print("\nModel setup successful! You can now run the Streamlit app.")
