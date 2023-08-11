from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

ps = PorterStemmer()
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

import pandas as pd
import numpy as np
full_form_data=pd.read_csv(r"shortCodesProduct.csv")
full_form_data=full_form_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

def replace_short_with_full(product_details, full_form_data):
    product_details = product_details.lower()
    short_to_full = dict(zip(full_form_data['ShortName'], full_form_data['Fullform']))
    words = product_details.split()
    updated_words = [short_to_full[word] if word in short_to_full else word for word in words]
    updated_product_details = ' '.join(updated_words)
    return updated_product_details


def cleanText(text):
    text = text.lower() 
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+|https\S+|www\S+|\S+\.com|\S+\.in|!', r'', text)
    text = re.sub(r'w.w.w+', r' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'[^\w\s.,-]', ' ', text)  # Remove all non-word characters except '.', ',', and '-'
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


def preprocess_text(text):
    words = word_tokenize(text)
    words_to_remove = ['qty', 'nos', 'no', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'etc', 'drg',
                       'mm', 'rdso', 'cbcb', 'mmt', 'mg', 'tc', 'sfc', 'm', 'sts', 'e',
                       'hrs', 'km', 'fdy', 'ft', 'q', 'alt', 'bt', 'pt', 'drgno', 'skel',
                       'aalpt', 'elw', 'bsl', 'wam', 'm', 'ml']
    words = [word for word in words if word not in words_to_remove or word == 'of']
    words = [word for word in words if len(word) > 1]
    words = [word for word in words if word not in stopwords]
    lemmatizedWords = ' '.join([lemmatizer.lemmatize(word) for word in words])
    return lemmatizedWords