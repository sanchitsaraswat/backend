from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Important!

model = joblib.load("model.pkl")

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
    app.run(debug=True)