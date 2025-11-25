import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS # Import the CORS extension

# --- 1. Initialization and Setup ---

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for all origins (CRITICAL FIX for browser connection)
CORS(app) 

# Global variables for model and scaler
model = None
scaler = None

# List of all 13 features the model expects (in the correct order)
# Note: 'monthly_rent_php' is the name the model was trained with.
# The frontend sends 'monthly_rent_php_log' but we fix that in the predict function.
FEATURE_ORDER = [
    'natural_lighting_score',
    'noise_level_db',
    'internet_speed_mbps',
    'room_occupancy_type',
    'study_space_availability',
    'bathroom_quality_rating',
    'kitchen_access_score',
    'security_level',
    'maintenance_quality',
    'distance_to_university_km',
    'public_transport_access',
    'monthly_rent_php', # The name the model expects
    'study_hours_available'
]

def load_assets():
    """Load the pre-trained model and scaler assets."""
    global model, scaler
    try:
        # Load the Random Forest Classifier model
        model = joblib.load('random_forest_model.joblib')
        
        # Load the StandardScaler (must be loaded as well)
        scaler = joblib.load('standard_scaler.joblib')
        
        print("Model and Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        # Optionally, raise the exception if the app should not run without the model
        # raise e 

# Load assets immediately when the application starts
load_assets()


# --- 2. Routes ---

@app.route('/')
def home():
    """Simple status check for the root path."""
    return "Student Environment Classifier API is running and ready for predictions at /predict!"


@app.route('/predict', methods=['POST'])
def predict():
    """Handles POST requests to make a prediction."""
    if model is None or scaler is None:
        return jsonify({
            'status': 'error',
            'message': 'Model assets not loaded. Check server logs.'
        }), 500

    try:
        # 1. Get the JSON data from the request
        data = request.get_json()
        
        # --- CRITICAL FIX: RENAME THE KEY TO MATCH THE TRAINED MODEL ---
        # The frontend sends 'monthly_rent_php_log', but the model needs 'monthly_rent_php'.
        if 'monthly_rent_php_log' in data:
            # Pop the old key and create a new key with the corrected name
            data['monthly_rent_php'] = data.pop('monthly_rent_php_log')
        # ------------------------------------------------------------------

        # 2. Convert the input dictionary into a pandas DataFrame (1 row)
        # Ensure the columns are in the exact order the scaler expects
        input_df = pd.DataFrame([data], columns=FEATURE_ORDER)
        
        # Check for missing features before scaling
        if input_df.isnull().values.any():
            missing_features = input_df.columns[input_df.isnull().any()].tolist()
            return jsonify({
                'status': 'error',
                'message': f'Missing data for the following required features: {", ".join(missing_features)}'
            }), 400

        # 3. Scale the input data
        scaled_input = scaler.transform(input_df)

        # 4. Make prediction
        prediction_score = int(model.predict(scaled_input)[0])

        # 5. Format the result based on the predicted score
        if prediction_score == 0:
            assessment = "Non-Conducive (Score 0)"
        elif prediction_score == 1:
            assessment = "Moderately Conducive (Score 1)"
        else: # Score 2
            assessment = "Highly Conducive (Score 2)"

        # 6. Return the result
        return jsonify({
            'status': 'success',
            'conduciveness_score': prediction_score,
            'assessment': assessment
        })

    except Exception as e:
        # Catch any other errors (e.g., incorrect number of features, data type issues)
        return jsonify({
            'status': 'error',
            'message': f'Internal processing error: {e}'
        }), 500


# To run the app locally (optional, Gunicorn is used in production)
if __name__ == '__main__':
    # Setting host='0.0.0.0' allows access from outside the container in deployment environments
    app.run(host='0.0.0.0', port=5000)