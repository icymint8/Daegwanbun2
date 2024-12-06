import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring
import CustomScoring as scoring
from nltk.corpus import stopwords

'''def getSearchEngineResult(query_dict):
    result_dict = {}
    ix = index.open_dir("./index")

    #with ix.searcher(weighting=scoring.BM25F()) as searcher:
    with ix.searcher(weighting=scoring.ScoringFunction()) as searcher:

        # TODO - Define your own query parser
        import nltk
        nltk.data.path.append('/venv/nltk_data')

        parser = QueryParser("contents", schema=ix.schema, group=OrGroup)
        stopWords = set(stopwords.words('english'))

        for qid, q in query_dict.items():
            new_q = ''
            for word in q.split(' '):
                if word.lower() not in stopWords:
                    new_q += word + ' '
            query = parser.parse(new_q.lower())

            results = searcher.search(query, limit=None)
            #print(results)
            result_dict[qid] = [result.fields()['docID'] for result in results]
    return result_dict'''

import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import nltk
import string

# NLTK setup
nltk.data.path.append('/venv/nltk_data')
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

stopWords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



def preprocess_text(text):
    query = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopWords]

    # Generate n-grams
    bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]
    trigrams = [' '.join(gram) for gram in ngrams(tokens, 3)]

    return tokens + bigrams + trigrams  # Combine unigrams, bigrams, and trigrams

def getSearchEngineResult(query_dict):
    """
    Processes queries, retrieves results, and returns ranked document IDs.
    """
    result_dict = {}
    ix = index.open_dir("./index")  # Open the index

    with ix.searcher(weighting=scoring.ScoringFunction()) as searcher:
    #with ix.searcher(weighting=scoring.BM25F()) as searcher:
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.9))  # Define query parser


        for qid, q in query_dict.items():
            # Preprocess query
            processed_query = preprocess_text(q)  # Generate 2-grams using preprocess_text
            #print(processed_query)
            # Join preprocessed terms for Whoosh query
            combined_query = ' '.join(processed_query)
            whoosh_query = parser.parse(combined_query,"title^2")
            #print(whoosh_query)
            # Search with the constructed query
            results = searcher.search(whoosh_query, limit=None)
            #print(results)
            # Collect document IDs
            result_dict[qid] = [result.fields()['docID'] for result in results]
            #result_dict[qid] = [result.score for result in results]
            #print(result_dict[qid][:3])
        return result_dict
