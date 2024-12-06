import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring
import CustomScoring as scoring
from nltk.corpus import stopwords


def getSearchEngineResult(query_dict):
    result_dict = {}
    ix = index.open_dir("./index")

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
    # with ix.searcher(weighting=scoring.ScoringFunction()) as searcher:

        # TODO - Define your own query parser
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup)
        stopWords = set(stopwords.words('english'))

        for qid, q in query_dict.items():
            new_q = ''
            for word in q.split(' '):
                if word.lower() not in stopWords:
                    new_q += word + ' '
            query = parser.parse(new_q.lower())
            results = searcher.search(query, limit=None)
            result_dict[qid] = [result.fields()['docID'] for result in results]

    return result_dict