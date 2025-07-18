from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)

# ✅ Allow frontend origin (replace with your actual frontend URL)
CORS(app, origins=["https://frontend-mp46.vercel.app"])

model = joblib.load("model.pkl")


@app.route("/",methods=["GET"])
def home():
    return "backend is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
        int(data["Age"]),
        int(data["Gender"]),
        int(data["Education_Level"]),
        int(data["Job_Title"]),
        int(data["Years_of_Experience"])
    ]
    prediction = model.predict([features])[0]
    return jsonify({"predicted_salary": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
