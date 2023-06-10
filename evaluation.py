from read_write_json import data_processing_to_json,convert_from_json ,convert_to_json,ids_queries_to_json


# ranking_results=convert_from_json('ranking_file.json')
# Compute the average precision (AP) for a single query




def compute_ap(ranking_results, qrels_results):
    # Compute the precision and recall at each rank
    precision = []
    recall = []
    relevant = 0
    for i, doc in enumerate(ranking_results):
        if doc in qrels_results:
            relevant += qrels_results[doc]
            precision.append(relevant / (i+1))
            recall.append(relevant / len(qrels_results))
    # Compute the average precision
    ap = sum(precision[i] * (recall[i] - recall[i-1]) for i in range(1, len(precision)) if recall[i] > recall[i-1])
    return ap

#########################################################################
def evaluation(ranking,qrels_results):
#    qrels_results=convert_from_json('antique_qrel.json')
   map_score = 0
   for query in ranking:
       if query in qrels_results:
         ap = compute_ap(ranking, qrels_results[query])
         map_score += ap
       else:
        continue
   map_score /= len(ranking)
#    print("MAP score:", map_score)



