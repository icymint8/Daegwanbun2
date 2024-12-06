'''import os.path
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, NUMERIC

schema = Schema(docID=NUMERIC(stored=True), contents=TEXT)
index_dir = "index"

if not os.path.exists(index_dir):
    os.makedirs(index_dir)

ix = create_in(index_dir, schema)

writer = ix.writer()

with open('./doc/document.text', 'r') as f:
    text = f.read()
    docs = text.split('********************************************\n')[:-1]
    for doc in docs:
        br = doc.find('Document')
        try:
            docID = int(doc[br:].split('\n')[0].split(' ')[-1])
        except:
            print(doc)
            print(br)
            raise ValueError
        doc_text = '\n'.join(doc[br:].split('\n')[1:])
        writer.add_document(docID=docID, contents=doc_text)

writer.commit()
print("make index!")'''

import os
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, NUMERIC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import nltk
import string

# NLTK setup
nltk.data.path.append('/venv/nltk_data')
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

stopWords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, use_ngrams=True):
    """
    Preprocesses text by:
    - Lowercasing
    - Tokenizing
    - Removing stopwords
    - Lemmatizing
    - Adding n-grams (optional)
    """

    p_text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(p_text.lower())  # Tokenize and lowercase
    tokens = [word for word in tokens if word not in stopWords]  # Remove stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize

    if use_ngrams:
        bigrams = ['_'.join(gram) for gram in ngrams(lemmatized_tokens, 2)]
        trigrams = ['_'.join(gram) for gram in ngrams(lemmatized_tokens, 3)]
        lemmatized_tokens.extend(bigrams + trigrams)

    return ' '.join(lemmatized_tokens)

# Schema definition
schema = Schema(title=TEXT(), docID=NUMERIC(stored=True), contents=TEXT())
index_dir = "index"

if not os.path.exists(index_dir):
    os.makedirs(index_dir)

ix = create_in(index_dir, schema)

writer = ix.writer()

# Read and preprocess documents
with open('./doc/document.text', 'r') as f:
    text = f.read()
    docs = text.split('********************************************\n')[:-1]
    for doc in docs[0:3]:
        br = doc.find('Document')
        try:
            docID = int(doc[br:].split('\n')[0].split(' ')[-1])  # Extract docID
            title = []
            for line in doc[br:].split('\n')[1:]:
                title.append(line)
                if not line:
                    break
            print(title)
        except ValueError:
            print(doc)
            print(br)
            raise

        doc_text = '\n'.join(doc[br:].split('\n')[1:])  # Extract document content
        processed_text = preprocess_text(doc_text)
        processed_title = preprocess_text(' '.join(title)) # Preprocess document
        writer.add_document(docID=docID, contents=processed_text, title=processed_title)  # Add to index

writer.commit()
print("Index created with preprocessed documents!")
