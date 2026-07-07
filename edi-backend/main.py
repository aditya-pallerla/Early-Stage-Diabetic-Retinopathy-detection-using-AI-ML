# edi-backend/main.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.model_service import run_full_prediction, TABULAR_FEATURES
from utils.history_service import get_all_history
import os
import io
import csv

# --- CONFIGURATION ---
app = Flask(__name__) # FIXED: __name__ syntax
# Allow frontend access
CORS(app) 
# API prefix (optional, but good practice)
API_PREFIX = "/api/v1" 

# ------------------------------
# Predict endpoint
# ------------------------------
@app.route(f"{API_PREFIX}/predict", methods=["POST"])
def predict_multimodal():
    try:
        # 1️⃣ Input Validation (Image and basic form data presence)
        if 'file' not in request.files:
            return jsonify({"error": "No retinal image file provided"}), 400
        
        image_file = request.files["file"]
        image_bytes = image_file.read() # Read image as bytes (in-memory)

        clinical_data = {
            # FIX: Use the Mixed-Case/Uppercase keys defined in FORM_KEY_MAP
            "HbA1c": request.form.get("HbA1c"),
            "fasting_glucose": request.form.get("fasting_glucose"),
            "cholesterol": request.form.get("cholesterol"),
            "duration_years": request.form.get("duration_years"),
            "age": request.form.get("age")
        }
        
        # 2️⃣ Run the full prediction service
        # This handles preprocessing, model inference, Grad-CAM, SHAP, and History saving.
        # We pass the image as bytes, not a file object.
        results = run_full_prediction(image_bytes, clinical_data) 

        # 3️⃣ Return the results to the frontend
        return jsonify(results), 200

    except ValueError as e:
        # Catch validation errors (e.g., non-numeric input)
        return jsonify({"error": f"Input Error: {str(e)}"}), 400
    except RuntimeError as e:
        # Catch model loading failure
        return jsonify({"error": f"Server Error: {str(e)}. Check ML assets."}), 503
    except Exception as e:
        # Catch all other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# ------------------------------
# Endpoint for fetching history
# ------------------------------
# @app.route(f"{API_PREFIX}/history", methods=["GET"])
# def get_prediction_history():
#     try:
#         history = get_all_history()
#         # The history_service now returns a list of dictionaries, ready for jsonify
#         return jsonify(history), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
@app.route(f"{API_PREFIX}/history", methods=["GET"])
def get_prediction_history():
    try:
        # Construct absolute path for safety
        base_dir = os.path.dirname(os.path.abspath(__file__))
        history_path = os.path.join(base_dir, "outputs", "history.csv")

        # Check if file exists
        if not os.path.exists(history_path):
            return jsonify([]), 200

        # Read CSV data
        history = []
        with open(history_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                history.append(row)

        return jsonify(history), 200

    except Exception as e:
        print("Error loading history:", e)
        return jsonify({"error": str(e)}), 500

# ------------------------------
if __name__ == "__main__": # FIXED: __name__ syntax
    app.run(debug=True, port=8000) # Run on port 8000 for standard API practice