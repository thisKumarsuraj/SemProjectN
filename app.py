from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
from flask import render_template


app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return render_template('index.html')


# Load your trained model
with open("random_forest_model (2).pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        # Extract features in correct order
        input_order = [
            'age', 'time_in_hospital', 'num_procedures', 'num_medications',
            'number_outpatient_log', 'number_emergency_log', 'number_inpatient_log',
            'number_diagnoses', 'metformin', 'repaglinide', 'nateglinide',
            'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'tolazamide', 'insulin', 'glyburide-metformin',
            'race_1', 'race_2', 'race_3', 'race_4', 'gender_1', 'admission_source_id_4',
            'admission_source_id_8', 'admission_source_id_9', 'admission_source_id_11',
            'discharge_disposition_id_2', 'discharge_disposition_id_7',
            'discharge_disposition_id_10', 'discharge_disposition_id_18',
            'max_glu_serum_1.0', 'A1Cresult_1',
            'primary_diag_1', 'primary_diag_2', 'primary_diag_3', 'primary_diag_4',
            'primary_diag_5', 'primary_diag_6', 'primary_diag_7', 'primary_diag_8'
        ]
        
        print("Received keys:", list(data.keys()))

        features = [data[key] for key in input_order]
        features_array = np.array([features])
        label = int(model.predict(features_array)[0])
        probability = float(model.predict_proba(features_array)[0][label])

        return jsonify({'label': label, 'probability': probability})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
