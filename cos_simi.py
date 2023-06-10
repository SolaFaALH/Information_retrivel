import math
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
#nltk.download('punkt')
from nltk.corpus import stopwords
#nltk.download('stopwords')
from collections import Counter
from nltk.tokenize import word_tokenize
from dateutil.parser import parse
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
# from spellchecker import Speller
from autocorrect import Speller

#########################################################################

def cosine_similarity2(query, index, data,ids):
    query_vector = {}
    document_vectors = {}
    query_length = 0

    # حساب تردد كلمات الاستعلام في المستندات المتوفرة
    for word in query:
        if word in index:
            if word in query_vector:
                query_vector[word] += 1
            else:
                query_vector[word] = 1
        else:
            continue
    

    # حساب تردد كلمات المستندات في المستندات المتوفرة وحساب طول كل متجه
    for word, postings in index.items():
        document_vectors[word] = {}
        for doc_id, count in postings:
            if word in query:  # التحقق من صحة معرف المستند
             document_vectors[word][doc_id] = count
             query_length += query_vector.get(word, 0)**2  # استخدام query_vector.get(word, 0) للحصول على قيمة افتراضية 0 إذا لم يتم العثور على الكلمة
            else:
             continue
    query_length = math.sqrt(query_length)
  
    # حساب قيمة cosine similarity بين الاستعلام والمستندات المتوفرة
    scores = {}
    for doc_id in range(len(data)):  # استخدام range(len(data)) بدلاً من data للحصول على المعرفات الفريدة
        if ids[doc_id] not in scores:
            score = 0
            document_length = 0
            for word in query:
                if word in document_vectors and ids[doc_id] in document_vectors[word]:
                    score += query_vector.get(word, 0) * document_vectors[word][ids[doc_id]]  # استخدام query_vector.get(word, 0) للحصول على قيمة افتراضية 0 إذا لم يتم العثور على الكلمة
                    document_length += document_vectors[word][ids[doc_id]] ** 2

            if document_length != 0:
                document_length = math.sqrt(document_length)
                score /= (query_length * document_length)
                scores[ids[doc_id]] = score

    return scores


#########################################################################
def ranking(scores):
    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    temp=[]
    for rank in ranked_results:
        temp.append(rank[0])
    return temp
