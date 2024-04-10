# flask server for topic modelling
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('topic_summary.csv')

# Convert the DataFrame to a dictionary for faster lookups
topic_info = df.set_index('topic').T.to_dict()


# Load the dictionary
dictionary = pickle.load(open('lda_dictionary.pkl', 'rb'))

app = Flask(__name__)
CORS(app)

stop_list = nltk.corpus.stopwords.words('english')
# The following list is to further remove some frequent words in the dataset.
stop_list += ['get','time', 'really', 'go', 'back', 'try', 'ordered', 'order',"a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "and", "or", "but", "so", "be", "have", "do", "would", "could", "like",
    "got", "took", "said", "I", "me", "my", "you", "u", "your", "it", "its", "they",
    "them", "their", "this", "that", "one", "also"]

# Create predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    best_lda = pickle.load(open('best_lda.pkl', 'rb'))
    
    # Get the data from the POST request.
    data = request.get_json()
    # get text from data
    document = data['text']
    
    # Preprocess the document
    preprocessed_doc = preprocess_text(document)

    # Convert the preprocessed document to BoW format
    bow_doc = dictionary.doc2bow(preprocessed_doc)

    # Get the topic distribution for the BoW document
    doc_topics = best_lda[bow_doc]

    # Sort the topic distribution by probability in descending order
    doc_topics_sorted = sorted(doc_topics, key=lambda x: x[1], reverse=True)


    # Convert the list of tuples to a list of dictionaries
    doc_topics_json = [{'id': id, 'probability': float(probability), 'top_ten_words': topic_info[id]['top_ten_words'], 'topic_summary': topic_info[id]['topic_summary']} for id, probability in doc_topics_sorted]

    return jsonify(doc_topics_json)


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Convert to lowercase
    tokens = [w.lower() for w in tokens]

    # Remove non-alphabetic tokens
    tokens = [w for w in tokens if re.search('^[a-z]+$', w)]

    # Remove stop words
    tokens = [w for w in tokens if w not in stop_list]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return tokens


if __name__ == '__main__':
    app.run(port=5007, debug=True)