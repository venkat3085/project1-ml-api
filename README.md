# 🌸 Iris Species Prediction API

A machine learning REST API that predicts Iris flower species based on sepal and petal measurements. Built with Flask and scikit-learn, deployed on AWS EC2.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Code Explanation](#code-explanation)
- [Deployment](#deployment)
- [Model Details](#model-details)

---

## 🎯 Overview

This project demonstrates a complete machine learning workflow from training to deployment:
1. Train a classification model on the Iris dataset
2. Create a REST API to serve predictions
3. Deploy to AWS EC2 for public access
4. Implement logging for monitoring

**Live Demo:** The API was deployed on AWS EC2 (now terminated for cost savings)

---

## ✨ Features

- ✅ Machine learning model with 100% accuracy
- ✅ RESTful API with JSON responses
- ✅ Browser-friendly documentation page
- ✅ Request/response logging
- ✅ Error handling
- ✅ Cloud deployment ready

---

## 🛠️ Technologies Used

- **Python 3.9+** - Programming language
- **scikit-learn** - Machine learning library
- **Flask** - Web framework
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **pickle** - Model serialization
- **AWS EC2** - Cloud hosting
- **Git/GitHub** - Version control

---

## 📁 Project Structure

```
project1-ml-api/
├── train_model.py          # Model training script
├── app.py                  # Flask API application
├── iris_model.pkl          # Trained model (not in git)
├── .gitignore             # Git ignore rules
├── mlops-project1-key.pem # SSH key (not in git)
└── README.md              # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/venkat3085/project1-ml-api.git
cd project1-ml-api
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn flask
```

4. **Train the model:**
```bash
python train_model.py
```

5. **Run the API:**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

---

## 💻 Usage

### Web Browser
Visit `http://localhost:5000` to see the API documentation page.

### Command Line (curl)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Python
```python
import requests

url = "http://localhost:5000/predict"
data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

response = requests.post(url, json=data)
print(response.json())  # {'species': 'setosa'}
```

---

## 📚 API Documentation

### Endpoints

#### `GET /`
Returns HTML documentation page with API usage instructions.

**Response:** HTML page

---

#### `POST /predict`
Predicts Iris species based on flower measurements.

**Request Body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "species": "setosa"
}
```

**Possible Species:**
- `setosa`
- `versicolor`
- `virginica`

**Error Response:**
```json
{
  "error": "error message"
}
```

---

## 🔍 Code Explanation

### train_model.py

This script trains the machine learning model.

```python
import pandas as pd
from sklearn.datasets import load_iris
```
- **pandas**: Library for data manipulation (DataFrames)
- **load_iris**: Function to load the Iris dataset

```python
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
```
- Loads the Iris dataset (150 samples, 4 features)
- Converts to pandas DataFrame for easy manipulation
- Adds species column (0=setosa, 1=versicolor, 2=virginica)

```python
df.head()      # First 5 rows
df.shape       # Dimensions (150, 5)
df.describe()  # Statistical summary
```
- Exploratory data analysis commands
- Helps understand the data before training

```python
X = df.drop('species', axis=1)  # Features (measurements)
y = df['species']                # Labels (species)
```
- **X**: Input features (4 measurements)
- **y**: Output labels (species to predict)
- Standard ML convention: X (uppercase), y (lowercase)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
- Splits data: 80% training, 20% testing
- **random_state=42**: Ensures reproducible results
- Training set: teaches the model
- Testing set: evaluates the model

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```
- Creates a Decision Tree classifier
- **fit()**: Trains the model on training data
- Learns patterns: "if petal_length < 2.5, then setosa"

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```
- Makes predictions on test data
- Compares predictions with actual values
- Calculates accuracy percentage

```python
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```
- Saves trained model to file
- **'wb'**: write in binary mode
- **pickle.dump()**: serializes Python object
- Allows reusing model without retraining

---

### app.py

This script creates the Flask API.

```python
import logging
from datetime import datetime
```
- **logging**: Records events (requests, errors)
- **datetime**: Timestamps for logs

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
```
- Configures logging system
- **level=INFO**: Log informational messages
- **format**: Timestamp + level + message
- **FileHandler**: Saves to api.log file
- **StreamHandler**: Prints to console

```python
app = Flask(__name__)
```
- Creates Flask application instance
- **__name__**: Tells Flask where to find resources

```python
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)
    logging.info("Model loaded successfully")
```
- Loads saved model from file
- **'rb'**: read in binary mode
- **pickle.load()**: deserializes model
- Logs success message

```python
@app.route('/', methods=['GET'])
def home():
    return html
```
- **@app.route()**: Decorator that maps URL to function
- **'/'**: Root URL (homepage)
- **methods=['GET']**: Accepts GET requests
- Returns HTML documentation page

```python
@app.route('/predict', methods=['POST'])
def predict():
```
- Maps /predict URL to predict function
- **POST**: Accepts data in request body
- Function runs when someone hits this endpoint

```python
try:
    data = request.get_json()
    logging.info(f"Received prediction request: {data}")
```
- **try**: Error handling block
- **get_json()**: Extracts JSON from request
- Logs the incoming request data

```python
features = np.array([[
    data['sepal_length'], 
    data['sepal_width'],
    data['petal_length'], 
    data['petal_width']
]])
```
- Extracts 4 measurements from request
- Converts to numpy array
- **[[...]]**: 2D array (1 sample, 4 features)
- Format model expects

```python
prediction = model.predict(features)
species = ['setosa', 'versicolor', 'virginica'][prediction[0]]
```
- Model predicts species (returns 0, 1, or 2)
- Converts number to species name
- **prediction[0]**: Gets first (only) prediction

```python
logging.info(f"Prediction: {species}")
return jsonify({'species': species})
```
- Logs the prediction
- **jsonify()**: Converts Python dict to JSON
- Returns JSON response to client

```python
except Exception as e:
    logging.error(f"Error during prediction: {str(e)}")
    return jsonify({'error': str(e)}), 400
```
- Catches any errors
- Logs error message
- Returns error response with 400 status code

```python
if __name__ == '__main__':
    logging.info("Starting Flask API server")
    app.run(debug=True, host='0.0.0.0', port=5000)
```
- Runs only if script executed directly
- **debug=True**: Shows detailed errors
- **host='0.0.0.0'**: Accessible from any IP
- **port=5000**: Runs on port 5000

---

## ☁️ Deployment

### AWS EC2 Deployment

1. **Launch EC2 Instance:**
```bash
aws ec2 run-instances \
  --image-id ami-08d7aabbb50c2c24e \
  --instance-type t3.micro \
  --key-name mlops-project1-key \
  --security-group-ids sg-XXXXXXXXX \
  --region us-east-1
```

2. **Connect via SSH:**
```bash
ssh -i mlops-project1-key.pem ec2-user@<PUBLIC_IP>
```

3. **Install Dependencies:**
```bash
sudo dnf update -y
sudo dnf install python3 python3-pip -y
pip3 install pandas numpy scikit-learn flask
```

4. **Transfer Files:**
```bash
scp -i mlops-project1-key.pem app.py ec2-user@<PUBLIC_IP>:~/
scp -i mlops-project1-key.pem iris_model.pkl ec2-user@<PUBLIC_IP>:~/
```

5. **Run API:**
```bash
python3 app.py
```

### Security Group Rules
- **Port 22**: SSH access
- **Port 5000**: Flask API access

---

## 📊 Model Details

### Dataset
- **Name:** Iris Dataset
- **Samples:** 150 (50 per species)
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Classes:** 3 (setosa, versicolor, virginica)

### Algorithm
- **Type:** Decision Tree Classifier
- **Library:** scikit-learn
- **Training Split:** 80/20
- **Accuracy:** 100% on test set

### Why Decision Tree?
- Easy to understand and interpret
- No feature scaling required
- Handles non-linear relationships
- Good for small datasets

---

## 📝 Logging

The API logs all requests and responses to:
- **Console:** Real-time monitoring
- **api.log file:** Persistent storage

**Log Format:**
```
2025-12-09 11:03:39 - INFO - Received prediction request: {...}
2025-12-09 11:03:39 - INFO - Prediction: setosa
```

---

## 🔒 Security Notes

- **Never commit** `.pem` files to Git
- **Never commit** API keys or secrets
- Use `.gitignore` to exclude sensitive files
- Use environment variables for configuration

---

## 🐛 Troubleshooting

### "Method Not Allowed" Error
- You're using GET instead of POST
- Use curl or API client for POST requests

### "Model file not found"
- Run `python train_model.py` first
- Ensure `iris_model.pkl` exists

### "Port already in use"
- Another process is using port 5000
- Kill the process or use a different port

---

## 🚀 Future Improvements

- [ ] Add input validation
- [ ] Implement model versioning
- [ ] Add unit tests
- [ ] Create Docker container
- [ ] Add CI/CD pipeline
- [ ] Implement authentication
- [ ] Add rate limiting
- [ ] Create Swagger documentation

---

## 📖 Learning Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Iris Dataset Info](https://archive.ics.uci.edu/ml/datasets/iris)
- [AWS EC2 Guide](https://docs.aws.amazon.com/ec2/)

---

## 👤 Author

**Venkat**
- GitHub: [@venkat3085](https://github.com/venkat3085)

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- Iris dataset by R.A. Fisher (1936)
- scikit-learn community
- Flask framework developers

---

**Built as part of MLOps learning journey - Project 1 of 12**
