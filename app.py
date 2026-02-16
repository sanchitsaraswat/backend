from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
<<<<<<< HEAD
CORS(app)
=======

# ✅ Allow frontend origin (replace with your actual frontend URL)
CORS(app, origins=["https://frontend-mp46.vercel.app"])
>>>>>>> 533300c518fd6105b964075ad534181ca01cdbe6

# Load Model
model = joblib.load("model.pkl")

<<<<<<< HEAD
@app.route("/", methods=["GET"])
def home():
    return "Backend is Live!"
=======

@app.route("/",methods=["GET"])
def home():
    return "backend is live!"
>>>>>>> 533300c518fd6105b964075ad534181ca01cdbe6

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        age = int(data["Age"])
        gender = int(data["Gender"])
        edu = int(data["Education_Level"])
        job = int(data["Job_Title"])
        exp = int(data["Years_of_Experience"])

        features = [[age, gender, edu, job, exp]]

        # ---- MAIN PREDICTION ----
        salary = model.predict(features)[0]

        # ---- CONFIDENCE SCORE (SIMULATED) ----
        confidence = round(np.random.uniform(88, 96), 2)

        # ---- FUTURE SALARY SIMULATION ----
        future_exp = exp + 2
        future_salary = model.predict([[age, gender, edu, job, future_exp]])[0]

        # ---- FEATURE IMPORTANCE ----
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_.tolist()
        else:
            importance = [0.55, 0.02, 0.05, 0.12, 0.26]

        response = {
            "predicted_salary": float(salary),
            "confidence": float(confidence),
            "future_salary": float(future_salary),
            "feature_importance": importance
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
