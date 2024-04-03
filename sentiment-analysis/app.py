# flask server for sentiment analysis
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import nltk
from nltk.util import ngrams
import contractions

# Load the dictionary
dictionary = pickle.load(open('dictionary.pickle', 'rb'))

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
    preprocess = preprocess_text(data['text'])

    # Make prediction using the model
    prediction = model.classify(preprocess)

    # Take the first value of prediction
    print(prediction)
    return jsonify(prediction)

def preprocess_text(text):
    # Preprocess the data
    stop_list = nltk.corpus.stopwords.words('english')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    sent = nltk.word_tokenize(text)
    
    # Lowercase conversion
    sent = [w.lower() for w in sent]
    
    # Stop word removal 
    sent = [w for w in sent if w not in stop_list]

    # Remove punctuation
    sent = [w for w in sent if w.isalnum()]
    
    # Lemmatization 
    sent = [lemmatizer.lemmatize(w) for w in sent]

    # Expand contractions
    sent = [contractions.fix(w) for w in sent]

    # Create bigrams
    bigrams = [' '.join(w) for w in list(ngrams(sent, 2))]
    
    sent.extend(bigrams)
    
    # Convert the original sentence into a vector.
    vector = dictionary.doc2bow(sent)
    
    # Create a dict object to store the document vector (in order to use NLTK's classifier later)
    sent_as_dict = {id:1 for (id, tf) in vector}
    return sent_as_dict

if __name__ == '__main__':
    app.run(port=5006, debug=True)

