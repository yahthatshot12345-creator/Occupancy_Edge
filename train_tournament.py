import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. Load Data
df = pd.read_csv('data/phase1_room_occupancy.csv')

# 2. data cleaning
X = df.drop(columns=['Date', 'Time', 'Room_Occupancy_Count'])
y = df['Room_Occupancy_Count']

# 3. Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% data vaulated for testing, 80% for training, random seed to ensure random sequency

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Comparison tournament
models = {
    "RandomForest": RandomForestRegressor(n_estimators=50, max_features='sqrt', random_state=42),
    "XGBoost": XGBRegressor(n_estimators=50),
    "LightGBM": LGBMRegressor(n_estimators=50, verbose=-1)
}

print("🚀 Starting Tournament...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test) # R-squared score
    print(f"🏆 {name} Accuracy: {score:.4f}")
    
    # Save the winner (we will save all 3 for the dashboard later)
    joblib.dump(model, f'models/{name.lower()}_model.pkl')

# Save the scaler so the dashboard uses the same math
joblib.dump(scaler, 'models/scaler.pkl')
print("\n✅ All models trained and saved to /models folder.")


# Model Ranking
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. Load model and  data structure
model = joblib.load('models/randomforest_model.pkl')
# Get column names from the original data (minus target/dates)
df = pd.read_csv('data/phase1_room_occupancy.csv')
features = df.drop(columns=['Date', 'Time', 'Room_Occupancy_Count']).columns

# 2. Extract Importance
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=features)

# 3. Print  Top 5
print("🥇 Top 5 Most Important Sensors for Random Forest Algorithm:")
print(feat_importances.nlargest(5))

# 4. Simple Plot 
feat_importances.nlargest(10).plot(kind='barh')
plt.title("What the AI is looking at")
plt.show()


# Three way comparison
import pandas as pd
import joblib

# 1. Load the models and feature names
rf = joblib.load('models/randomforest_model.pkl')
xgb = joblib.load('models/xgboost_model.pkl')
lgbm = joblib.load('models/lightgbm_model.pkl')

df = pd.read_csv('data/phase1_room_occupancy.csv')
features = df.drop(columns=['Date', 'Time', 'Room_Occupancy_Count']).columns

# 2. Extract Importance (Note: different models call them different things)
importance_df = pd.DataFrame({
    'Feature': features,
    'RandomForest': rf.feature_importances_,
    'XGBoost': xgb.feature_importances_,
    'LightGBM': lgbm.feature_importances_ / lgbm.feature_importances_.sum() # Normalize LightGBM to 100%
})

# 3. Sort by Random Forest and display
print("Feature Importance Comparison:")
print(importance_df.sort_values(by='RandomForest', ascending=False).to_string(index=False))