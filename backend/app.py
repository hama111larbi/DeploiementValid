from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import pandas as pd
import tensorflow as tf
import traceback
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configuration de TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
else:
    print("Aucun GPU détecté, utilisation du CPU")

# Chargement des modèles
try:
    print("Chargement des modèles...")
    
    # Modèle de coût
    cost_model = joblib.load('cost_model.joblib')
    cost_scaler = joblib.load('cost_scaler.joblib')
    print("Modèle de coût chargé")
    
    # Modèle de mortalité
    mortality_model = joblib.load('model.joblib')
    mortality_scaler = joblib.load('scaler.joblib')
    mortality_features = joblib.load('feature_info.joblib')
    print("Modèle de mortalité chargé")
    
    # Modèle de stroke
    try:
        stroke_model = tf.keras.models.load_model('Bi_Nexus.h5', compile=False)
        stroke_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Modèle de stroke chargé")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle de stroke : {str(e)}")
        traceback.print_exc()
        raise
    
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {str(e)}")
    traceback.print_exc()
    raise

@app.route('/predict_cost', methods=['POST'])
def predict_cost():
    try:
        data = request.json
        print("Données reçues pour coût:", data)

        # Créer un DataFrame avec les noms exacts des caractéristiques du modèle
        input_data = pd.DataFrame([{
            'Resolution_Duration_Days': float(data['resolution_days']),
            'years_experience': float(data['years_experience']),
            'Lits': float(data['beds_number']),
            'Pourcentage_Lits': float(data['beds_percentage'])
        }])

        features_scaled = cost_scaler.transform(input_data)
        prediction = cost_model.predict(features_scaled)[0]

        return jsonify({'cost': float(prediction)})

    except Exception as e:
        print("Erreur prédiction coût:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict_mortality():
    try:
        data = request.json
        print("Données reçues pour mortalité:", data)

        input_data = pd.DataFrame([{
            'BMI': float(data['bmi']),
            'gender': int(data['gender']),
            'years_experience': float(data['years_experience']),
            'speciality': int(data['speciality']),
            'discharge_location': int(data['discharge_location'])
        }])

        input_data = input_data[mortality_features['names']]
        features_scaled = mortality_scaler.transform(input_data)
        prediction_prob = mortality_model.predict_proba(features_scaled)[0][1]
        prediction = 1 if prediction_prob >= 0.6 else 0

        risk_factors = []
        if prediction == 1:
            if float(data['bmi']) > 30:
                risk_factors.append("IMC élevé")
            if float(data['years_experience']) < 5:
                risk_factors.append("Médecin peu expérimenté")
            if int(data['discharge_location']) == 3:
                risk_factors.append("Lieu de sortie à risque")

        return jsonify({
            'prediction': int(prediction),
            'probability': float(prediction_prob),
            'risk_factors': risk_factors
        })

    except Exception as e:
        print("Erreur prédiction mortalité:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict-stroke', methods=['POST'])
def predict_stroke():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400

        file = request.files['image']
        if not file.filename:
            return jsonify({'error': 'Fichier invalide'}), 400

        # Vérifier le type de fichier
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Format de fichier non supporté. Utilisez PNG ou JPG'}), 400

        try:
            # Ouvrir et prétraiter l'image
            img = Image.open(file.stream)
            img = img.convert('RGB')  # Convertir en RGB si nécessaire
            img = img.resize((224, 224), Image.LANCZOS)
            
            # Convertir en array et normaliser
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Faire la prédiction
            prediction = stroke_model.predict(img_array, verbose=0)
            probability = float(prediction[0][0])
            result = 'Stroke' if probability > 0.5 else 'Normal'

            # Convertir l'image pour l'affichage
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return jsonify({
                'prediction': result,
                'probability': probability,
                'image': img_str
            })

        except Exception as img_error:
            print("Erreur lors du traitement de l'image:", str(img_error))
            traceback.print_exc()
            return jsonify({'error': 'Erreur lors du traitement de l\'image'}), 400

    except Exception as e:
        print("Erreur prédiction stroke:", str(e))
        traceback.print_exc()
        return jsonify({'error': 'Une erreur inattendue est survenue'}), 500

if __name__ == '__main__':
    app.run(debug=True)
