from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join("model", "insurance_model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        # Example features: adjust according to your model
        age = float(data["age"])
        bmi = float(data["bmi"])
        children = float(data["children"])
        smoker = 1 if data["smoker"].lower() == "yes" else 0

        features = np.array([[age, bmi, children, smoker]])
        prediction = model.predict(features)[0]

        return jsonify({"prediction": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
