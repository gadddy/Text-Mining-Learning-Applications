import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import re


# Load the dictionary
dictionary = pickle.load(open('lda_dictionary.pkl', 'rb'))


stop_list = nltk.corpus.stopwords.words('english')
# The following list is to further remove some frequent words in the dataset.
stop_list += ['get','time', 'really', 'go', 'back', 'try', 'ordered', 'order',"a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "and", "or", "but", "so", "be", "have", "do", "would", "could", "like",
    "got", "took", "said", "I", "me", "my", "you", "u", "your", "it", "its", "they",
    "them", "their", "this", "that", "one", "also"]


text = "An excellent choice for Hong Kong style roast pork, roast duck, and char siu (BBQ pork). Super affordable too. Service is quick and the ambiance is entirely what you'd expect for this kind of tasty and delicious meal. That said if you want the absolute best roast duck in the world, this isn't it. But you also can't be too greedy considering how cheap this place is."

def predict(data):
    # Load the model
    model = pickle.load(open('best_lda.pkl', 'rb'))
    # Get the data from the POST request.
    # Make prediction using the model
    input = preprocess_text(data)
    # Convert the preprocessed document to BoW format
    bow_doc = dictionary.doc2bow(input)

    # Get the topic distribution for the BoW document
    doc_topics = model[bow_doc]

    return doc_topics



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

result = predict(text)
print(result)