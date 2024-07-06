'''
Author: Karan Vasudevamurthy
UTA ID: 1002164438
Net ID: kxv4439

Functions:
- `preprocess_documents(corpusroot)`: Preprocess all documents in the given directory.
- `getidf(token, perform_stemming=True)`: Computes the inverse document frequency (IDF) for a given token.
- `getweight(filename, token)`: Computes the TF-IDF weight for a given token in a specified document.
- `compute_document_vectors()`: Precomputes the TF-IDF vectors and normalization factors for each document.
- `query(qstring)`: Processes a query string, computes its vector, and returns the most relevant document based on cosine similarity.

Usage:
- By default, running this script will execute the queries mentioned in the assignment.
- If you would like change the default queries, you can change them in the outputs section. (Line 125)
'''

import os
import math
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import traceback
from time import perf_counter

# Initialize tokenizer, stemmer, stopwords, and data structures
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
documents = {}
term_doc_freq = {}
total_docs = 0
doc_vectors = {}
doc_norm_factors = {}

# Set the path to the directory containing the files
corpusroot = './US_Inaugural_Addresses'

def preprocess_documents(corpusroot):
    global total_docs
    if not os.path.exists(corpusroot) or not os.path.isdir(corpusroot):
        raise ValueError('Path error. Please check the path given in corpusroot')
    
    for filename in os.listdir(corpusroot):
        if filename.endswith('.txt'):
            with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as file:
                doc = file.read().lower()
                tokens = tokenizer.tokenize(doc)
                filtered_tokens = [token for token in tokens if token not in stop_words]
                stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
                
                documents[filename] = stemmed_tokens
                total_docs += 1
                
                unique_terms = set(stemmed_tokens)
                for term in unique_terms:
                    if term not in term_doc_freq:
                        term_doc_freq[term] = 0
                    term_doc_freq[term] += 1

def getidf(token, perform_stemming=True):
    if perform_stemming:
        token = stemmer.stem(token)
    
    if token in term_doc_freq:
        return math.log10(total_docs / term_doc_freq[token])
    else:
        return 0

def getweight(filename, token):
    stemmed_token = stemmer.stem(token)
    if filename not in documents:
        return 0

    tokens = documents[filename]
    term_freq = tokens.count(stemmed_token)
    
    if term_freq == 0:
        return 0

    tf = 1 + math.log10(term_freq)
    idf = getidf(token)
    tf_idf = tf * idf

    sum_squares = 0
    for t in set(tokens):
        tf_t = 1 + math.log10(tokens.count(t))
        idf_t = getidf(t, perform_stemming=False)
        sum_squares += (tf_t * idf_t) ** 2
    
    norm_factor = math.sqrt(sum_squares)
    return tf_idf / norm_factor if norm_factor != 0 else 0

def compute_document_vectors():
    for filename, tokens in documents.items():
        term_freqs = {token: tokens.count(token) for token in set(tokens)}
        doc_weights = {token: (1 + math.log10(freq)) * getidf(token, perform_stemming=False) for token, freq in term_freqs.items()}
        doc_norm_factor = math.sqrt(sum(weight ** 2 for weight in doc_weights.values()))
        doc_vectors[filename] = {token: weight / doc_norm_factor for token, weight in doc_weights.items()}
        doc_norm_factors[filename] = doc_norm_factor

def query(qstring):
    query_tokens = tokenizer.tokenize(qstring.lower())
    filtered_query_tokens = [token for token in query_tokens if token not in stop_words]
    stemmed_query_tokens = [stemmer.stem(token) for token in filtered_query_tokens]

    term_freqs = {token: stemmed_query_tokens.count(token) for token in set(stemmed_query_tokens)}
    query_weights = {token: (1 + math.log10(freq)) for token, freq in term_freqs.items()}
    query_norm_factor = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
    query_vector = {token: weight / query_norm_factor for token, weight in query_weights.items()}

    scores = {}
    for filename, document_vector in doc_vectors.items():
        dot_product = sum(query_vector[token] * document_vector.get(token, 0) for token in query_vector.keys())
        scores[filename] = dot_product

    if scores:
        best_match = max(scores, key=scores.get)
        return (best_match, scores[best_match])
    else:
        return None

try:
    start = perf_counter()

    preprocess_documents(corpusroot)
    compute_document_vectors()

    # Outputs
    print("%.12f" % getidf('democracy'))
    print("%.12f" % getidf('foreign'))
    print("%.12f" % getidf('states'))
    print("%.12f" % getidf('honor'))
    print("%.12f" % getidf('great'))
    print("--------------")
    print("%.12f" % getweight('19_lincoln_1861.txt', 'constitution'))
    print("%.12f" % getweight('23_hayes_1877.txt', 'public'))
    print("%.12f" % getweight('25_cleveland_1885.txt', 'citizen'))
    print("%.12f" % getweight('09_monroe_1821.txt', 'revenue'))
    print("%.12f" % getweight('37_roosevelt_franklin_1933.txt', 'leadership'))
    print("--------------")
    print("(%s, %.12f)" % query("states laws"))
    print("(%s, %.12f)" % query("war offenses"))
    print("(%s, %.12f)" % query("british war"))
    print("(%s, %.12f)" % query("texas government"))
    print("(%s, %.12f)" % query("world civilization"))
    end = perf_counter()
    print(f"The program took: {end-start:.3f} seconds")
except Exception:
    print(traceback.format_exc())
