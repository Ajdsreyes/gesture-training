from flask import Flask, request, jsonify
from feature_extraction import extract_features
import numpy as np
import joblib

app = Flask(__name__)

# Load trained SVM model
model = joblib.load("gesture_svm_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # Get gesture sequence
    gesture = data["touch_position"]

    # Extract features
    features = extract_features(gesture)

    # Convert to model input
    features = np.array(features).reshape(1, -1)

    # Predict gesture
    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)