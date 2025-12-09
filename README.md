# 🌸 Iris Species Prediction API

Machine learning REST API that predicts Iris flower species. Built with Flask and scikit-learn, deployed on AWS EC2.

## 🛠️ Technologies

- Python 3.9+
- scikit-learn (ML)
- Flask (API)
- pandas & numpy
- AWS EC2

## 📁 Project Structure

```
project1-ml-api/
├── train_model.py     # Model training
├── app.py            # Flask API
├── iris_model.pkl    # Trained model
└── README.md         # Documentation
```

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/venkat3085/project1-ml-api.git
cd project1-ml-api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn flask

# Train model
python train_model.py

# Run API
python app.py
```

## 💻 Usage

**Request:**
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

**Response:**
```json
{
  "species": "setosa"
}
```

**Possible species:** setosa, versicolor, virginica

## 📚 Code Explanation

### train_model.py

```python
# Load Iris dataset (150 samples, 4 features, 3 species)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Split features (X) and labels (y)
X = df.drop('species', axis=1)  # Measurements
y = df['species']                # Species to predict

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)  # 100%

# Save model to file
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

**Key concepts:**
- **train_test_split**: Prevents overfitting by testing on unseen data
- **fit()**: Trains the model on training data
- **predict()**: Makes predictions on new data
- **pickle**: Saves model to reuse without retraining

### app.py

```python
# Setup logging (saves to api.log and console)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# Load saved model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Extract measurements from request
    data = request.get_json()
    features = np.array([[
        data['sepal_length'], 
        data['sepal_width'],
        data['petal_length'], 
        data['petal_width']
    ]])
    
    # Make prediction
    prediction = model.predict(features)
    species = ['setosa', 'versicolor', 'virginica'][prediction[0]]
    
    return jsonify({'species': species})
```

**Key concepts:**
- **@app.route()**: Maps URL to function
- **methods=['POST']**: Accepts POST requests with data
- **request.get_json()**: Extracts JSON from request
- **jsonify()**: Converts Python dict to JSON response

## ☁️ AWS Deployment

```bash
# Launch EC2 instance (t3.micro, free tier)
aws ec2 run-instances \
  --image-id ami-08d7aabbb50c2c24e \
  --instance-type t3.micro \
  --key-name mlops-project1-key \
  --security-group-ids sg-XXXXXXXXX

# Connect via SSH
ssh -i mlops-project1-key.pem ec2-user@<PUBLIC_IP>

# Install dependencies
sudo dnf update -y
sudo dnf install python3 python3-pip -y
pip3 install pandas numpy scikit-learn flask

# Transfer files
scp -i mlops-project1-key.pem app.py ec2-user@<PUBLIC_IP>:~/
scp -i mlops-project1-key.pem iris_model.pkl ec2-user@<PUBLIC_IP>:~/

# Run API
python3 app.py
```

**Security Group:** Allow ports 22 (SSH) and 5000 (API)

## 📊 Model Details

- **Dataset:** Iris (150 samples, 4 features, 3 classes)
- **Algorithm:** Decision Tree Classifier
- **Accuracy:** 100% on test set
- **Training Split:** 80/20

## 📝 Logging

Logs saved to `api.log` and console:
```
2025-12-09 11:03:39 - INFO - Received prediction request: {...}
2025-12-09 11:03:39 - INFO - Prediction: setosa
```

## 🔒 Security

- `.pem` files excluded via `.gitignore`
- Never commit secrets or API keys
- Use environment variables for sensitive config

## 👤 Author

**Venkat** - [@venkat3085](https://github.com/venkat3085)

---

**Part of MLOps learning journey - Project 1 of 12**
