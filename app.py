import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# --- Configuration: Feature List and Class Map ---
# !!! IMPORTANT: These feature names and their order MUST match the data 
# your model was trained on in the Colab notebook. 
FEATURE_COLS = [
    'natural_lighting_score', 'noise_level_db', 'internet_speed_mbps', 
    'room_occupancy_type', 'study_space_availability', 'bathroom_quality_rating', 
    'kitchen_access_score', 'security_level', 'maintenance_quality', 
    'distance_to_university_km', 'public_transport_access', 'monthly_rent_php_log', 
    'study_hours_available'
]

CLASS_MAP = {
    0: "Not Conducive / Low Performance 📉",
    1: "Moderately Conducive / Medium Performance ➡️",
    2: "Highly Conducive / High Performance 🌟"
}

# --- API Setup ---
app = Flask(__name__)

# Load model and scaler once at startup to improve prediction speed
try:
    MODEL = joblib.load('random_forest_model.joblib')
    SCALER = joblib.load('scaler.joblib')
    print("Model and Scaler loaded successfully.")
except Exception as e:
    # If loading fails, it indicates a critical error (missing files, etc.)
    print(f"ERROR: Could not load model or scaler. Error: {e}")
    
# --- API Endpoint Definition ---

@app.route('/', methods=['GET'])
def home():
    """Simple health check."""
    return "Student Environment Classifier API is running and ready for predictions at /predict!"

@app.route('/predict', methods=['POST'])
def predict():
    """Receives JSON data, preprocesses it, and returns a classification."""
    
    # Force get_json ensures it works even if the content type isn't perfectly set
    data = request.get_json(force=True)
    
    try:
        # 1. Convert to DataFrame to ensure the correct column structure for the scaler
        input_df = pd.DataFrame([data], columns=FEATURE_COLS)
        
        # 2. Apply the pre-trained Standard Scaler
        # This transforms the new data using the mean/std from the training set
        scaled_features = SCALER.transform(input_df)
        
        # 3. Get Prediction
        prediction_class = int(MODEL.predict(scaled_features)[0])
        result = CLASS_MAP.get(prediction_class, "Prediction Error")
        
        # 4. Return response
        return jsonify({
            'conduciveness_score': prediction_class,
            'assessment': result,
            'status': 'success'
        })

    except KeyError:
        # If any of the 13 required features are missing
        return jsonify({'error': 'Missing required features in the input JSON.', 'status': 'fail'}), 400
    except Exception as e:
        # Generic error handler
        return jsonify({'error': f'Internal processing error: {str(e)}', 'status': 'fail'}), 500

if __name__ == '__main__':
    # Running locally on port 5000 with auto-reloading (debug=True)
    app.run(debug=True, port=5000)
    
    