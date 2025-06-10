
from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('iris_model.pkl')
target_names = ['setosa', 'versicolor', 'virginica']

app = Flask(__name__)

@app.route('/')
def home():
    return "Iris model Flask API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    input_features = np.array(features).reshape(1, -1)
    prediction = model.predict(input_features)
    predicted_class = target_names[prediction[0]]
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
