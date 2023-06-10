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
# from spellchecker import SpellChecker
from autocorrect import Speller
import json
from collections import defaultdict
#from inverted_index import calculate_idf, calculate_tf, calculate_tf_idf, inverted_index
import math



# def correct_words(data):
#  spell = Speller(lang='en')
#  Query = spell(data)
#  return Query
nlp = spacy.load('en_core_web_sm')
#########################################################################
def correct_spelling(data):
    spell = Speller(lang='en')
    corrected_data = []
    for sublist in data:
        corrected_sublist = [spell(text) for text in sublist]
        corrected_data.append(corrected_sublist)
    return corrected_data
#########################################################################
def convert_lower_case(data): 
    if isinstance(data, str):  # التحقق من أن القيمة هي نص
        return data.apply(lambda x: x.lower())
    else:
        return data
    
#########################################################################

def tokenize(data):
    return [word_tokenize(str(text).lower()) for text in data.tolist()]


#########################################################################
def remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for doc in data:
      filtered_doc = [word for word in doc if not word in stop_words]
      filtered_words.append(filtered_doc)
    return filtered_words
    
#########################################################################
def stemming(data):
    stemmer = PorterStemmer()
    stemmed_words = []
    for doc in data:
      stemmed_doc = [stemmer.stem(word) for word in doc]
      stemmed_words.append(stemmed_doc)
    return stemmed_words

#########################################################################
def remove_punctuation(data):
    normalized_words = []
    for doc in data:
        normalized_doc = [contractions.fix(word).replace("’", "'") for word in doc]
        normalized_doc = [''.join(char for char in word if char not in string.punctuation + ' "''_-~`\\:.') for word in normalized_doc]
        normalized_doc = list(filter(None, normalized_doc))
        normalized_words.append(normalized_doc)
    return normalized_words

#########################################################################


def lemmatization(data):
    lemmatized_words = []
    for doc in data:
        lemmatized_doc = [token.lemma_ for token in nlp(" ".join(doc))]
        lemmatized_words.append(lemmatized_doc)
    return lemmatized_words


#########################################################################

def replace_abbreviations(data):
    abbreviations = {
        "usa": "united states of america",
        "us": "united states",
        "uk": "united kingdom",
        "uae": "united arab emirates",
        "canada": "canada",
        "aus": "australia",
        "ger": "germany",
        "fr": "france",
        "spain": "spain",
        "italy": "italy",
        "japan": "japan",
        "china": "china",
        "india": "india",
        "brazil": "brazil",
        "mexico": "mexico",
        "russia": "russia",
        "egypt": "egypt",
        "sa": "saudi arabia",
        "turkey": "turkey",
        "iran": "iran",
        "iraq": "iraq",
        "syria": "syria",
        "lebanon": "lebanon",
        "jordan": "jordan",
        "qatar": "qatar",
        "kuwait": "kuwait",
        "bahrain": "bahrain",
        "oman": "oman",
        "yemen": "yemen",
        "pakistan": "pakistan",
        "afghanistan": "afghanistan",
        "indonesia": "indonesia",
        "malaysia": "malaysia",
        "philippines": "philippines",
        "thailand": "thailand",
        "vietnam": "vietnam",
        "south korea": "south korea",
        "north korea": "north korea",
        "argentina": "argentina",
        "buenos aires": "buenos aires",
        "chile": "chile",
        "peru": "peru",
        "colombia": "colombia",
        "venezuela": "venezuela",
        "ecuador": "ecuador",
        "bolivia": "bolivia",
        "paraguay": "paraguay",
        "uruguay": "uruguay",
        "south africa": "south africa",
        "nigeria": "nigeria",
        "egypt": "egypt",
        "morocco": "morocco",
        "kenya": "kenya",
        "ethiopia": "ethiopia",
        "ghana": "ghana",
        "uganda": "uganda",
        "tanzania": "tanzania",
        "algeria": "algeria",
        "angola": "angola",
        "mozambique": "mozambique",
        "zimbabwe": "zimbabwe",
        "can": "canada",
        "aus": "australia",
        "ger": "germany",
        "fr": "france",
        "esp": "spain",
        "ita": "italy",
        "jpn": "japan",
        "chn": "china",
        "ind": "india",
        "bra": "brazil",
        "mex": "mexico",
        "rus": "russia",
        "egy": "egypt",
        "sau": "saudi arabia",
        "tur": "turkey",
        "irn": "iran",
        "irq": "iraq",
        "syr": "syria",
        "lbn": "lebanon",
        "jor": "jordan",
        "qat": "qatar",
        "kwt": "kuwait",
        "bhr": "bahrain",
        "omn": "oman",
        "yem": "yemen",
        "pak": "pakistan",
        "afg": "afghanistan",
        "idn": "indonesia",
        "mys": "malaysia",
        "phl": "philippines",
        "tha": "thailand",
        "vnm": "vietnam",
        "kor": "south korea",
        "arg": "argentina",
        "chl": "chile",
        "per": "peru",
        "col": "colombia",
        "ven": "venezuela",
        "ecu": "ecuador",
        "bol": "bolivia",
        "pry": "paraguay",
        "ury": "uruguay",
        "zaf": "south africa",
        "nga": "nigeria",
        "mar": "morocco",
        "ken": "kenya",
        "eth": "ethiopia",
        "gha": "ghana",
        "uga": "uganda",
        "tza": "tanzania",
        "dza": "algeria",
        "ago": "angola",
        "moz": "mozambique",
        "zwe": "zimbabwe",
         "can't": "cannot",
        "govt": "government",
        "intl": "international",
        "dept": "department",
        "phd": "doctor of philosophy",
        "bldg": "building",
        "edu": "education",
        "info": "information",
        "corp": "corporation",
        "co": "company",
        "mr": "mister",
        "mrs": "missus",
        "dr": "doctor",
        "jr": "junior",
        "sr": "senior",
        "inc": "incorporated",
        "mgr": "manager",
        "univ": "university",
        "ph": "phone",
        "est": "established",
        "acct": "account",
        "adm": "administration",
        "agcy": "agency",
        "assn": "association",
        "asst": "assistant",
        "attn": "attention",
        "bros": "brothers",
        "ca": "california",
        "dept": "department",
        "dist": "district",
        "div": "division",
        "ed": "editor",
        "eng": "engineer",
        "esp": "especially",
        "estd": "established",
        "feb": "february",
        "gov": "government",
        "i.e.": "that is",
        "jan": "january",
        "lib": "library",
        "lit": "literature",
        "math": "mathematics",
        "med": "medical",
        "min": "minimum",
        "mtg": "meeting",
        "natl": "national",
        "nov": "november",
        "oct": "october",
        "op": "operation",
        "org": "organization",
        "pub": "public",
        "rec": "recreation",
        "rep": "representative",
        "res": "research",
        "sec": "secretary",
        "sept": "september",
        "spec": "special",
        "stmt": "statement",
        "supv": "supervisor",
        "tech": "technology",
        "temp": "temporary",
        "univ": "university",
        "util": "utility",
        "wks": "weeks",
        "wrk": "work",
        "yr": "year",
        "jan.": "january",
        "feb.": "february",
        "mar.": "march",
        "apr.": "april",
        "jun.": "june",
        "jul.": "july",
        "aug.": "august",
        "sept.": "september",
        "oct.": "october",
        "nov.": "november",
        "dec.": "december",
        "january.": "january",
        "february.": "february",
        "march.": "march",
        "april.": "april",
        "june.": "june",
        "july.": "july",
        "august.": "august",
        "september.": "september",
        "october.": "october",
        "november.": "november",
        "december.": "december",
        "am": "morning",
        "pm": "afternoon/evening",
        "a.m.": "morning",
        "p.m.": "afternoon/evening",
        "gmt": "greenwich mean time",
        "est": "eastern standard time",
        "edt": "eastern daylight time",
        "cst": "central standard time",
        "cdt": "central daylight time",
        "mst": "mountain standard time",
        "mdt": "mountain daylight time",
        "pst": "pacific standard time",
        "pdt": "pacific daylight time",
        "utc": "coordinated universal time",
         "can’t": "can not",
        "won’t": "will not",
        "shouldn’t": "should not",
        "didn’t": "did not",
        "it’s": "it is",
        "I’m": "I am",
        "you’re": "you are",
        "he’s": "he is",
        "she’s": "she is",
        "we’re": "we are",
        "they’re": "they are",
        "isn’t": "is not",
        "aren’t": "are not",
        "wasn’t": "was not",
        "weren’t": "were not",
        "hasn’t": "has not",
        "haven’t": "have not",
        "doesn’t": "does not",
        "don’t": "do not",
        "didn’t": "did not",
        "should’ve": "should have",
        "could’ve": "could have",
        "would’ve": "would have",
        "might’ve": "might have",
        "mustn’t": "must not",
        "needn’t": "need not",
        "ought to": "should",
        "gotta": "got to",
        "wanna": "want to",
        "kinda": "kind of",
        "sorta": "sort of",
        "gonna": "going to",
        "lemme": "let me",
        "gimme": "give me",
        "ain’t": "am not",
        "aren’t": "are not",
        "can’t": "cannot",
        "could’ve": "could have",
        "couldn’t": "could not",
        "didn’t": "did not",
        "doesn’t": "does not",
        "don’t": "do not",
        "hadn’t": "had not",
        "hasn’t": "has not",
        "haven’t": "have not",
        "he’d": "he would",
        "he’ll": "he will",
        "he’s": "he is",
        # يمكنك إضافة المزيد من الاختصارات وتعويضاتها هنا
    }

    replaced_words = []
    for doc in data:
        replaced_doc = [abbreviations.get(word, word) for word in doc]
        replaced_words.append(replaced_doc)
    return replaced_words

    