from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import base64

app = Flask(__name__)
CORS(app)

# Load model and threshold
model = tf.keras.models.load_model('Bi_Nexus.h5')
# Using default threshold of 0.5 for binary classification
THRESHOLD = 0.5

def preprocess_image(image_bytes):
    # Convert bytes to image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize
    img = img.resize((224, 224))
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/api/predict-stroke', methods=['POST'])
def predict_stroke():
    try:
        # Get the image from the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Preprocess the image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        
        # Apply threshold
        is_stroke = bool(prediction > THRESHOLD)
        
        # Convert image to base64 for display
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return jsonify({
            'prediction': 'Stroke' if is_stroke else 'Normal',
            'probability': float(prediction),
            'threshold': THRESHOLD,
            'image': image_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
