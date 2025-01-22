from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the ML model and scaler
model = pickle.load(open("air_quality.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return "Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse form data
        data = request.form

        Temperature = float(data.get('Temperature'))
        Humidity = float(data.get('Humidity'))
        PM25 = float(data.get('PM25'))
        PM10 = float(data.get('PM10'))
        SO2 = float(data.get('SO2'))
        NO2 = float(data.get('NO2'))
        CO = float(data.get('CO'))
        Proximity = float(data.get('Proximity'))
        Population = float(data.get('Population'))

        PM = PM25 - PM10
        features = pd.DataFrame([[Temperature, Humidity, PM, NO2, SO2, CO, Proximity, Population]], 
                                columns=['Temperature', 'Humidity', 'PM', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density'])

            
            
        features_scaled = scaler.transform(features)
        scaled_features_df = pd.DataFrame(features_scaled, columns=features.columns)

        # Predict using the model
        prediction = model.predict(scaled_features_df)
        prediction = int(prediction[0])

        # Interpret the result
        result = "Air quality is good" if prediction == 1 else "Air quality is poor"

        # Return the prediction result
        return jsonify({"Prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
