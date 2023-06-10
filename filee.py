from asyncio import log
import pandas as pd
import nltk
import string
import math
import re
import contractions
import spacy
import dateparser
import os
import numpy as np
from nltk.stem import PorterStemmer
from collections import defaultdict
from dateutil.parser import parse
#nltk.download('punkt')
nltk.download('wordnet')
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
import json
from collections import defaultdict
from process_text import convert_lower_case ,correct_spelling , tokenize ,remove_stop_words,stemming,remove_punctuation,lemmatization,replace_abbreviations
from query import  convert_query_lower_case ,query_tokenization,remove_query_stop_words,remove_query_punctuation,query_lemmatization,query_stemming,replace_query_abbreviations
from inverted_index import inverted_index
from cos_simi import cosine_similarity2 ,ranking
from read_qrel import load_qrel_file
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from read_write_json import data_processing_to_json,convert_from_json ,convert_to_json,ids_queries_to_json
from evaluation import evaluation

from gensim.models import Word2Vec
from gensim.matutils import Sparse2Corpus
from gensim.models import TfidfModel






antique_df = pd.read_csv('antique-collection.txt', sep='\t', header=None, names=['id','text_right'])
antique_sample = antique_df['text_right'].head(30000)
# print(antique_sample)

wikir_df = pd.read_csv('wikIR1k/documents.csv')
wikir_sample = wikir_df['text_right'].head(100)
# print(wikir_sample)

antique_query = pd.read_csv('antique-train-queries.txt', sep='\t', header=None, names=['id','text_right'])
antique_query_sample = antique_query['text_right'].head(100)
# print(antique_query_sample)


# ids_queries_to_json(antique_query['id'],antique_query['text_right'])
ids_queries=convert_from_json('ids_queries.json')
# print(ids_queries['What #\'s double is greater than its half by 51?'])

wikir_query = pd.read_csv('wikIR1k/test/queries.csv')
wikir_query_sample = wikir_query['text_left'].head(100)
#print(wikir_query_sample)

dataset_id_chosen=antique_df['id'].head(30000)

#antique_sample#wikir_sample#antique_query_sample#wikir_query_sample
data=convert_lower_case(antique_sample)
data=tokenize(data)
data=remove_stop_words(data)
data =replace_abbreviations(data) 
data=remove_punctuation(data)
data = lemmatization(data)
data =stemming(data)
# data = correct_spelling(data)
# print(data)

data_processing_to_json(dataset_id_chosen,data)

query1 = "What #'s double is greater than its half by 51?"
# query1 ="small iraq rita"
tokens = convert_query_lower_case(query1)
tokens = query_tokenization(tokens)
tokens = remove_query_stop_words(tokens)
tokens = remove_query_punctuation(tokens)
tokens = query_lemmatization(tokens)
tokens = replace_query_abbreviations(tokens)
tokens = query_stemming(tokens)
# print(tokens)


inverted_index=inverted_index(data,dataset_id_chosen)
# print(inverted_index)
data1=convert_from_json('dataset_processing.json')
scores=cosine_similarity2(tokens,inverted_index,data1,dataset_id_chosen)
# print(scores)
ranking =ranking(scores)

convert_to_json(ranking, 'ranking_file.json')
ranking11=convert_from_json('ranking_file.json')
antique_qrel_file=convert_from_json('antique_qrel.json')
temp=antique_qrel_file[ids_queries[query1]]
# print(temp)
# for item in temp:
#     print(item)
#     print(temp[item])
ll=[]
# print(ranking11)
evaluation(ranking11,temp)
# for item in ranking:
#     for qr in temp:
#         if item[0]==qr:
#          ll.append(temp[item])
#         else:
#             continue
# print(ll)         

    

#قرينا ال qrel وخولنا ل json وقرينا من ال json
# qrel= load_qrel_file('antique-train.qrel.txt')
# convert_to_json(qrel,'antique_qrel.json')
# temp=convert_from_json('antique_qrel.json')
# print(temp["2531329"])
# print(qrel)

def retrieve_documents(tokens, inverted_index):
    retrieved_documents = set()
    for token in tokens:
        if token in inverted_index:
            for doc_id in inverted_index[token]:
                documents = [entry[0] for entry in inverted_index[token]]
                retrieved_documents.update(documents)
    return retrieved_documents


retrieved_documents = retrieve_documents(tokens, inverted_index)
# print(retrieved_documents)
dataset_processing=convert_from_json('dataset_processing.json')


# Data Representation
def represent_text(data):
    vectorizer = TfidfVectorizer()
    x=vectorized_data = vectorizer.fit_transform(data)
    return vectorizer, x

#تحويل القائمة المتداخلة إلى جملة نصية واحدة لكل قائمة فرعية
sentences = [' '.join(dataset_processing[id]) for id in retrieved_documents]
vectorizer,representation = represent_text(sentences)
# print(representation)

# model = Word2Vec(data, min_count=1)

antique = Sparse2Corpus(representation, documents_columns=False)
tfidf_model = TfidfModel(antique)
antique_tfidf = tfidf_model[antique]

#window :5 كلمات قبل وبعد الكلمة
model = Word2Vec(sentences, vector_size=100, window=5, min_count=3,compute_loss=False)
model.train(antique_tfidf, total_examples=len(sentences), epochs=30)


# save the model to a file
model_path = 'antique_model'
if os.path.exists(model_path):
    os.remove(model_path)
model.save(model_path)

model = Word2Vec.load('antique_model')

# تمثيل جملة الاستعلام باستخدام Word2Vec
query_words = [' '.join(sublist) for sublist in tokens]
query_vector = model.wv[query_words]

# تحويل التمثيل المتجهي إلى تمثيل TF-IDF gensim
query_gen = dictionary.doc2bow(query_words)
query_tfidf = tfidf_model[query_gen]

# استخراج النص الأقرب لجملة الاستعلام
sims = sorted(tfidf_model[query_gen], key=lambda item: -item[1])[:5]
most_similar_texts = [df.iloc[sim[0]]['text'] for sim in sims]
print(most_similar_texts)


