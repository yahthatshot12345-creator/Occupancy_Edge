import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # Switched to Classifier

# 1. Load Phase 2 Data
df = pd.read_csv('data/phase2_room_occupancy.csv')

# 2. Create the 15-Minute "Future" Target
# Predicting if the room is occupied (1) or empty (0) 15 minutes from now
shift_minutes = 15
df['Future_Occupancy'] = df['Occupancy'].shift(-shift_minutes)
df_forecast = df.dropna().copy()

# 3. Define Features (Note: We drop the 'current' occupancy to avoid leakage)
features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
X = df_forecast[features]
y = df_forecast['Future_Occupancy']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_p2 = StandardScaler()
X_train_scaled = scaler_p2.fit_transform(X_train)
X_test_scaled = scaler_p2.transform(X_test)

# 5. Edge-Optimized Classifier
model_p2 = RandomForestClassifier(n_estimators=50, max_features='sqrt', random_state=42)
model_p2.fit(X_train_scaled, y_train)

# 6. Save
joblib.dump(model_p2, 'models/phase2_forecaster.pkl')
joblib.dump(scaler_p2, 'models/scaler_phase2.pkl')

print(f"🚀 Phase 2 Accuracy: {model_p2.score(X_test_scaled, y_test):.4f}")