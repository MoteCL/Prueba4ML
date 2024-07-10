from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)
model_path = os.path.join(os.path.dirname(__file__), '../models/model.h5')
tokenizer_path = os.path.join(os.path.dirname(__file__), '../models/tokenizer.pkl')

model = load_model(model_path)

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['input']
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)
    # spam = bool(prediction[0][0] > 0.5)  # Convert numpy.bool_ to Python bool
    result = "Spam" if bool(prediction[0][0] > 0.5) else "Not Spam"

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
