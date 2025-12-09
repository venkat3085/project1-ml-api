from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# Create Flask app
app = Flask(__name__)

# Load the trained model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)
    logging.info("Model loaded successfully")

@app.route('/', methods=['GET'])
def home():
    html = """
    <html>
    <head><title>Iris ML API</title></head>
    <body style="font-family: Arial; max-width: 800px; margin: 50px auto;">
        <h1>🌸 Iris Species Prediction API</h1>
        <p>Machine Learning API for predicting Iris flower species</p>
        
        <h2>API Endpoint</h2>
        <p><strong>POST</strong> /predict</p>
        
        <h2>Example Request</h2>
        <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px;">
curl -X POST http://3.226.74.231:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
        </pre>
        
        <h2>Example Response</h2>
        <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px;">
{
  "species": "setosa"
}
        </pre>
        
        <h2>Species</h2>
        <ul>
            <li>setosa</li>
            <li>versicolor</li>
            <li>virginica</li>
        </ul>
        
        <p style="color: #666; margin-top: 50px;">
            Built with Flask, scikit-learn, deployed on AWS EC2
        </p>
    </body>
    </html>
    """
    return html

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        logging.info(f"Received prediction request: {data}")
        
        features = np.array([[data['sepal_length'], data['sepal_width'], 
                              data['petal_length'], data['petal_width']]])
        
        # Make prediction
        prediction = model.predict(features)
        species = ['setosa', 'versicolor', 'virginica'][prediction[0]]
        
        logging.info(f"Prediction: {species}")
        return jsonify({'species': species})
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    logging.info("Starting Flask API server")
    app.run(debug=True, host='0.0.0.0', port=5000)
