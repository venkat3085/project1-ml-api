from flask import Flask, request, jsonify
import pickle
import numpy as np
#Create Flask app
app = Flask(__name__)

#Load the trained model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    features = np.array([[data['sepal_length'], data['sepal_width'],
                          data['petal_length'], data['petal_width']]])

    # Make prediction
    prediction = model.predict(features)
    species = ['setosa', 'versicolor', 'virginica'][prediction[0]]

    return jsonify({'species': species})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
