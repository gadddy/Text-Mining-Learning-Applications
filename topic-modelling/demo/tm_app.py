# flask server for topic modelling
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import nltk
from nltk.util import ngrams
import contractions

# Load the dictionary
dictionary = pickle.load(open('', 'rb'))

app = Flask(__name__)
CORS(app)

# Create predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model = pickle.load(open('sentiment_classifier.pkl', 'rb'))
    # Get the data from the POST request.
    data = request.get_json()
    # Make prediction using the model
    preprocess = preprocess_text()


if __name__ == '__main__':
    app.run(port=5006, debug=True)