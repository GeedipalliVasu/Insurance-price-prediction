from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)

# ------------------- Model Loading -------------------
model_path = "model/insurance_model.pkl"
if not os.path.exists(model_path):
    print("Training model for the first time...")

    data = pd.read_csv("insurance.csv")
    le = LabelEncoder()
    data["sex"] = le.fit_transform(data["sex"])
    data["smoker"] = le.fit_transform(data["smoker"])
    data["region"] = le.fit_transform(data["region"])

    X = data.drop("charges", axis=1)
    y = data["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, model_path)
    print("✅ Model trained and saved.")
else:
    print("Loading pre-trained model...")
    model = joblib.load(model_path)

# ------------------- Routes -------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        age = int(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        # Encode categorical manually (same order as training)
        sex_dict = {"female": 0, "male": 1}
        smoker_dict = {"no": 0, "yes": 1}
        region_dict = {"northwest": 1, "northeast": 0, "southeast": 2, "southwest": 3}

        x = [[
            age,
            sex_dict.get(sex.lower(), 0),
            bmi,
            children,
            smoker_dict.get(smoker.lower(), 0),
            region_dict.get(region.lower(), 0),
        ]]

        prediction = model.predict(x)[0]
        prediction_text = f"💰 Estimated Insurance Cost: ${prediction:,.2f}"

        return render_template("index.html", prediction_text=prediction_text)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
