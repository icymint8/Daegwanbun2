import QueryResult
import evaluate
query_dict = evaluate.readQueryFile('doc/query.text')
#print(query_dict)
dict_query = QueryResult.getSearchEngineResult(query_dict)
k=evaluate.getGroundtruthRelevance(query_dict.keys())
for key in dict_query.keys():
    print(key,dict_query[key][:10],k[key])
#print(evaluate.getSearchEngineResult(dict_query[key])))
