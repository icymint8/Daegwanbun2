import numpy as np
from QueryResult import getSearchEngineResult


def readQueryFile(filename):
    query_dict = {}

    with open(filename, 'r') as f:
        text = f.read()
        queries = text.split(' #\n')
        for query in queries[:-1]:
            br = query.find('\n')
            qID = int(query[:br])
            q = query[br+1:].replace('\n',' ')
            query_dict[qID] = q

    return query_dict


def getGroundtruthRelevance(query_ids):
    relevant_dict = {}
    start_idx = 0
    with open('./doc/relevance.text', 'r') as f:
        all_items = [item for item in f.read().split(' ') if item!='' and item!='\n']
        while start_idx<len(all_items):
            queryID = int(all_items[start_idx])
            num_related_docs = int(all_items[start_idx+1])
            docIds = [int(all_items[idx]) for idx in range(start_idx+2, num_related_docs+start_idx+2)]
            if queryID in query_ids:
                # add into relevant_dict
                relevant_dict[queryID] = docIds
            start_idx += num_related_docs+2

    return relevant_dict


def evaluate(query_dict, relevent_dict, results_dict):
    BPREF = []

    for queryID in query_dict.keys():
        relevantCount = 0
        nonRelevantCount = 0
        score = 0
        results = results_dict[queryID]
        relevantDocuments = relevent_dict[queryID]
        relDocCount = len(relevantDocuments)

        for document in results:
            if document in relevantDocuments:
                relevantCount += 1
                if nonRelevantCount >= relDocCount:
                    score += 0
                else:
                    score += (1 - nonRelevantCount / relDocCount)
            else:
                nonRelevantCount += 1
            if relevantCount == relDocCount:
                break
        score = score / relDocCount
        BPREF.append(score)

    print(np.mean(BPREF))
    print(BPREF)
if __name__ == '__main__':
    query_dict = readQueryFile('doc/query.text')
    relevant_dict = getGroundtruthRelevance(query_dict.keys())
    results_dict = getSearchEngineResult(query_dict)
    evaluate(query_dict, relevant_dict, results_dict)
