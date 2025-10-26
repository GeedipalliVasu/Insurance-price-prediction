# train_and_save.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create model directory
os.makedirs("model", exist_ok=True)

# 1️⃣ Load dataset
df = pd.read_csv("data/insurance.csv")

# 2️⃣ Encode categorical columns
encoders = {}
for col in ["sex", "smoker", "region"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 3️⃣ Split data
X = df.drop("charges", axis=1)
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Train model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test_scaled)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 7️⃣ Save model, scaler, encoders
joblib.dump(model, "model/insurance_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")

print("✅ Model, scaler, and encoders saved successfully!")
