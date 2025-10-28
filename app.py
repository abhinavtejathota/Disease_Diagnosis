from flask import Flask, jsonify, request, send_from_directory
from feature import prepare_features
import joblib, os, pandas as pd

app = Flask(__name__, template_folder='.')

MODELS = {}
for name in ["heart", "stroke", "hepatitis"]:
    path = f"Trained_Models/{name}_model.pkl"
    if os.path.exists(path):
        MODELS[name] = joblib.load(path)
        print(f"✅ Loaded {path}")
    else:
        print(f"⚠️ Model not found: {path}")


@app.route("/")
def home():
    return send_from_directory("./templates", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        disease = data.get("disease")
        features = data.get("features")

        if not disease or disease not in MODELS:
            return jsonify({"error": f"Invalid disease: {disease}"}), 400

        df = prepare_features(disease, features)
        df = df.fillna(0)

        model = MODELS[disease]
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else None

        result = {
            "prediction": int(pred),
            "probability": round(float(prob), 3) if prob is not None else None,
        }

        label = (
            f"🩸 Positive (High Risk) — Probability: {result['probability']}"
            if result["prediction"] == 1
            else f"✅ Negative (Low Risk) — Probability: {result['probability']}"
        )

        return jsonify({
            "prediction_text": label,
            "raw": result,
            "input_preview": df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
