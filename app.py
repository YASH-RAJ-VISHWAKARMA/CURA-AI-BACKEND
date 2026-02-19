import os
from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from flask_cors import CORS
from utils_data import load_and_clean, get_symptom_list
from generate_mapping import generate_doctor_mapping

# ------------ CONFIG ------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "disease_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
MAPPING_CSV = os.path.join(BASE_DIR, "disease_doctor_mapping.csv")

# ------------ FLASK APP ------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "supersecretkey"
CORS(app)

# ------------ LOAD MODEL & DATA ------------
model = tf.keras.models.load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

train_df, test_df = load_and_clean()
symptom_columns = get_symptom_list(train_df)

if os.path.exists(MAPPING_CSV):
    mapping_df = pd.read_csv(MAPPING_CSV, header=None)
    doctor_mapping = {row[0]: row[1] for _, row in mapping_df.iterrows()}
else:
    doctor_mapping = generate_doctor_mapping(train_df['prognosis'].unique())

# ------------ USERS (dummy login) ------------
USERS = {
    "test@example.com": "1234",
    "yash@example.com": "password"
}

# ------------ PREDICTION HELPER ------------
def predict_disease_from_symptoms(user_symptoms):
    input_data = np.zeros(len(symptom_columns), dtype=np.float32)
    for s in user_symptoms:
        if s in symptom_columns:
            input_data[symptom_columns.index(s)] = 1

    prediction = model.predict(input_data.reshape(1, -1), verbose=0)[0]
    top_idxs = prediction.argsort()[-3:][::-1]
    top_diseases = label_encoder.inverse_transform(top_idxs)
    top_confidences = (prediction[top_idxs] * 100).round(2)

    results = []
    for d, c in zip(top_diseases, top_confidences):
        doc = doctor_mapping.get(d, "General Physician")
        results.append({"disease": str(d), "confidence": float(c), "doctor": doc})
    return results

# ------------ ROUTES ------------
@app.route("/")
def home():
    return render_template("index.html", symptoms=symptom_columns)

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email, password = data.get("email"), data.get("password")
    if email in USERS and USERS[email] == password:
        session["user"] = email
        return jsonify({"success": True, "message": "Logged in!"})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"success": True, "message": "Logged out"})

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json(force=True)
#     raw = data.get("message", "")

#     if isinstance(raw, str):
#         raw_inputs = [raw]
#     elif isinstance(raw, list):
#         raw_inputs = raw
#     else:
#         return jsonify({"error": "Invalid input format"}), 400

#     responses = []
#     for entry in raw_inputs:
#         user_symptoms = [s.strip().replace(" ", "_") for s in entry.split(",") if s.strip()]

#         if not user_symptoms:
#             responses.append({"input": entry, "error": "No symptoms provided"})
#             continue

#         valid_symptoms = [s for s in user_symptoms if s in symptom_columns]

#         if not valid_symptoms:
#             responses.append({"input": entry, "error": "No valid symptoms recognized"})
#             continue

#         results = predict_disease_from_symptoms(valid_symptoms)
#         responses.append({"input": entry, "results": results})

#     if len(responses) == 1:
#         return jsonify(responses[0])
#     return jsonify({"responses": responses})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    raw = data.get("message", "")

    if isinstance(raw, str):
        raw_inputs = [raw]
    elif isinstance(raw, list):
        raw_inputs = raw
    else:
        return jsonify({"error": "Invalid input format"}), 400

    responses = []
    for entry in raw_inputs:
        # 🔽 Normalize: lowercase + replace spaces with underscores
        user_symptoms = [
            s.strip().lower().replace(" ", "_")
            for s in entry.split(",")
            if s.strip()
        ]

        if not user_symptoms:
            responses.append({"input": entry, "error": "No symptoms provided"})
            continue

        # 🔽 Match against lowercase version of symptom_columns
        valid_symptoms = [s for s in user_symptoms if s in symptom_columns]

        if not valid_symptoms:
            responses.append({"input": entry, "error": "No valid symptoms recognized"})
            continue

        results = predict_disease_from_symptoms(valid_symptoms)
        responses.append({"input": entry, "results": results})

    if len(responses) == 1:
        return jsonify(responses[0])
    return jsonify({"responses": responses})



# ------------ ENTRY POINT ------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    # app.run(debug=True)