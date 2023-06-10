import pandas as pd
import nltk
import string
import math
import re
import contractions
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
from autocorrect import Speller
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
import json
from collections import defaultdict
#########################################################################
def inverted_index(data, ids):
    index = defaultdict(list)
    for i, doc in enumerate(data):
        counted_words = set()
        for word in doc:
            if isinstance(word, float):  # التحقق من نوع القيمة
                continue
            if word not in counted_words:
                index[word].append((ids[i], data[i].count(word)))
                counted_words.add(word)
    return index



