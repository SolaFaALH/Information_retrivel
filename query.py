import nltk
import string
import re
import contractions
import spacy
import numpy as np
from nltk.stem import PorterStemmer
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from spellchecker import SpellChecker
from autocorrect import Speller

# # تحميل قائمة الكلمات الداخلة ضمن قائمة الـ stop words باللغة الإنجليزية
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# تحميل قائمة الـ stopwords والقوائم الإضافية المطلوبة من بيئة الـ spaCy
stop_words = set(stopwords.words('english'))

# تحميل نموذج spaCy
nlp = spacy.load("en_core_web_sm")



def convert_query_lower_case(data):
    return data.lower()

def query_tokenization(data):
    # تنفيذ عملية الـ tokenization باستخدام word_tokenize
    tokens = word_tokenize(data)
    return tokens

def remove_query_stop_words(data):
    # إزالة الـ stopwords من الكلمات الموجودة في الاستعلام
    filtered_words = [word for word in data if not word in stop_words]
    return filtered_words

def remove_query_punctuation(tokens):
    # نمط العلامات الخاصة التي تحتاج لإزالتها
    pattern = r'[\'_~`\\:.)($#?!,."\'+=\&]'

    # إزالة العلامات الخاصة
    cleaned_tokens = [re.sub(pattern, '', token) for token in tokens]
    # إزالة العناصر الفارغة من القائمة
    cleaned_tokens = list(filter(None, cleaned_tokens))
    
    return cleaned_tokens

def query_lemmatization(data):
    # تنفيذ عملية الـ lemmatization باستخدام spaCy
    lemmatized_doc = [token.lemma_ for token in nlp(" ".join(data))]
    
    return lemmatized_doc

def query_stemming(data):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in data]
    return stemmed_tokens

def replace_query_abbreviations(data):
    abbreviations = {
        # قائمة الاختصارات والاستبدالات الخاصة بك
        "don't": "do not",
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
        # إضافة المزيد من الاختصارات والاستبدالات هنا
    }

    # استبدال الاختصارات في النص
    replaced_doc = [abbreviations.get(word, word) for word in data]
    
    return replaced_doc


