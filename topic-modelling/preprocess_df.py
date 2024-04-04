import pandas as pd
import nltk
import re
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer

stop_list = nltk.corpus.stopwords.words('english')
# The following list is to further remove some frequent words in the dataset.
stop_list += ['get','time', 'really', 'go', 'back', 'try', 'ordered', 'order',"a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "and", "or", "but", "so", "be", "have", "do", "would", "could", "like",
    "got", "took", "said", "I", "me", "my", "you", "u", "your", "it", "its", "they",
    "them", "their", "this", "that", "one", "also"]

def load_corpus(df, column):
    # df is a DataFrame and column is the name of the column with text data.
    corpus = df[column].tolist()  # Convert the column to a list of texts.
    return corpus

def corpus2docs(corpus):
    # Initialize spell checker
    spell = SpellChecker()
    lemmatizer = WordNetLemmatizer()
    # corpus is a list of documents.
    docs1 = [nltk.word_tokenize(doc) for doc in corpus]
    docs2 = [[w.lower() for w in doc] for doc in docs1]
    docs3 = [[w for w in doc if re.search('^[a-z]+$', w)] for doc in docs2]
    #correct spelling mistakes
    #if word has a spelling mistake, use spell.correction
    #if word is correct, don't change it
    #if word is not in the dictionary, don't change it
    #below line has issues with computation time
    #docs5 = [[spell.correction(w) if w not in spell else w for w in doc] for doc in docs3]

    docs4 = [[w for w in doc if w not in stop_list] for doc in docs3]
    # Add lemmatization
    docs5 = [[lemmatizer.lemmatize(w) for w in doc] for doc in docs4]
    return docs5

def docs2vecs(docs, dictionary):
    # docs is a list of documents returned by corpus2docs.
    # dictionary is a gensim.corpora.Dictionary object.
    vecs1 = [dictionary.doc2bow(doc) for doc in docs]
    return vecs1

