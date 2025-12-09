from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

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
