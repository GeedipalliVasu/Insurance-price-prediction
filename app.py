# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load saved files
model = joblib.load("model/insurance_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/encoders.pkl")

def preprocess(age, sex, bmi, children, smoker, region):
    sex_encoded = encoders["sex"].transform([sex])[0]
    smoker_encoded = encoders["smoker"].transform([smoker])[0]
    region_encoded = encoders["region"].transform([region])[0]
    row = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    row_scaled = scaler.transform(row)
    return row_scaled

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        processed_input = preprocess(age, sex, bmi, children, smoker, region)
        prediction = model.predict(processed_input)[0]
        return render_template("index.html", prediction_text=f"Predicted Insurance Cost: ${prediction:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
