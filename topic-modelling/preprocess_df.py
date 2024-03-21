import pandas as pd
import nltk
import re
from spellchecker import SpellChecker

stop_list = nltk.corpus.stopwords.words('english')
# The following list is to further remove some frequent words in SGNews.
stop_list += ['would', 'said', 'say', 'year', 'day', 'also', 'first', 'last', 'one', 'two', 'people', 'told', 'new', 'could', 'singapore', 'three', 'may', 'like', 'world', 'since']

import gensim

def load_corpus(df, column):
    # df is a DataFrame and column is the name of the column with text data.
    corpus = df[column].tolist()  # Convert the column to a list of texts.
    return corpus

def corpus2docs(corpus):
    # Initialize spell checker
    spell = SpellChecker()
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
    return docs4

def docs2vecs(docs, dictionary):
    # docs is a list of documents returned by corpus2docs.
    # dictionary is a gensim.corpora.Dictionary object.
    vecs1 = [dictionary.doc2bow(doc) for doc in docs]
    return vecs1