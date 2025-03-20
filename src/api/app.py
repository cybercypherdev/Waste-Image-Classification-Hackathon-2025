from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import cv2
from PIL import Image
import io
import joblib
import markdown

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 64x64
    resized = cv2.resize(gray, (64, 64))
    # Flatten and normalize
    flattened = resized.flatten() / 255.0
    return flattened

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read the image file
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Scale the features
        scaled_features = scaler.transform([processed_image])
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Get the probability of the predicted class
        probability = max(probabilities)
        
        # Convert prediction to string
        prediction_str = 'Recyclable' if prediction == 1 else 'Organic'
        
        return jsonify({
            'prediction': prediction_str,
            'probability': float(probability),
            'predictions': {
                'Recyclable': float(probabilities[1]),
                'Organic': float(probabilities[0])
            }
        })

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Failed to process image'}), 500

@app.route('/readme', methods=['GET'])
def get_readme():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': 'Failed to load README'}), 500

@app.route('/images', methods=['GET'])
def get_images():
    # Return some sample images for the carousel
    images = [
        {'url': 'https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?w=800&auto=format&fit=crop&q=60'},
        {'url': 'https://images.unsplash.com/photo-1526951521999-620a0084b512?w=800&auto=format&fit=crop&q=60'},
        {'url': 'https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?w=800&auto=format&fit=crop&q=60'}
    ]
    return jsonify(images)

if __name__ == '__main__':
    app.run(debug=True, port=8000) 