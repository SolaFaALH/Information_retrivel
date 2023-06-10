from asyncio import log
import pandas as pd
import nltk
import string
import math
import re
import contractions
import spacy
import dateparser
import numpy as np
from nltk.stem import PorterStemmer
from collections import defaultdict
from dateutil.parser import parse

#قراءة ملف ال QREL وأخذ القيم منه
def load_qrel_file(file_path):
    qrel_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            query_id, _, doc_id, relevance = line.strip().split()
            query_id = int(query_id)
            doc_id = int(doc_id)
            relevance = int(relevance)
            if query_id not in qrel_data:
                qrel_data[query_id] = {}
            qrel_data[query_id][doc_id] = relevance
    return qrel_data

