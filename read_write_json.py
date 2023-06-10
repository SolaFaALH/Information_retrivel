from collections import defaultdict
import json

##########################################################################################
def convert_to_json(temp,name_of_file):
   file_to_json = json.dumps(temp)
   with open(name_of_file, 'w') as f:
    f.write(file_to_json)
    f.close()

##########################################################################################
def convert_from_json(name_of_file):
   with open(name_of_file, 'r') as f:
    file_from_json = f.read()
    return json.loads(file_from_json)
   
##########################################################################################

def data_processing_to_json(ids,data):
    data_set=defaultdict(list)
    for i, doc in enumerate(data):
        data_set[ids[i]]=doc
    convert_to_json(data_set,'dataset_processing.json')


############################################################################################


def ids_queries_to_json(ids,data):
    data_set=defaultdict(list)
    for i, doc in enumerate(data):
        data_set[doc]=str(ids[i])
    convert_to_json(data_set,'ids_queries.json')

