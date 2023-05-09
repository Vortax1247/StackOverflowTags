from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from bs4 import BeautifulSoup 
from sklearn.metrics import hamming_loss


spacy.load('en_core_web_sm')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

from nltk.corpus import stopwords

stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']

def text_extract(row) :
    blocklist = [  'code','pre','a']
    text_elements = [t for t in BeautifulSoup(row).find_all(
    string=True) if t.parent.name not in blocklist and t!='\n']
    text = str()
    for i in range(len(text_elements)) :
        text = text+text_elements[i]
    return text

def sent_to_words(sentences):
    yield(gensim.utils.simple_preprocess(str(sentences), deacc=True))
    return sent_to_words(sentences)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_w] for doc in texts]

def lemmatization_spacy(texts, allowed_postags=['NOUN','PROPN']):
    texts_out = []
    for sent in texts :
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def transform_bow_lem_spacy_fct(desc_text) :
    word_tokens = list(sent_to_words(desc_text))
    sw = remove_stopwords(word_tokens)
    lem_w = lemmatization_spacy(sw)
    transf_desc_text = ' '.join(lem_w[0])
    return transf_desc_text


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def print_score(y_test,y_pred ):
    print("Hamming loss: {}".format(hamming_loss(y_test,y_pred )))
    print("Hamming score: {}".format(hamming_score(y_test,y_pred )))
    print("---")    

def transform_text_bow_lem_spacy_fct(desc_text) :
    text = text_extract(desc_text)
    word_tokens = list(sent_to_words(text))

    lem_w = lemmatization_spacy(word_tokens)
    sw = remove_stopwords(lem_w)
    transf_desc_text = ' '.join(lem_w[0])
    return transf_desc_text
def text_processing(df) : 
    X = df['Title'].apply(transform_bow_lem_spacy_fct)+' '+df['Body'].apply(transform_text_bow_lem_spacy_fct)
    return X

