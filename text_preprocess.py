from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from bs4 import BeautifulSoup 

import time
import logging

logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

spacy.load('en_core_web_sm')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def text_extract(row) : 
    blocklist = [  'code','pre','a']
    text_elements = [t for t in BeautifulSoup(row).find_all(
    string=True) if t.parent.name not in blocklist and t!='\n']
    text = str()
    for i in range(len(text_elements)) : 
        text = text+text_elements[i]
    return text

def Tags_to_list (row) :
    Tags_list =[]
    for j in range(5):
            if row['Tag_'+str(j)] is not np.nan :
                Tags_list.append(row['Tag_'+str(j)])
    return Tags_list

def Tags_extract(df) :
    dict_tags = dict()
    df['Len_tags'] = int()
    for i,row in enumerate(df.Tags) :
        tags  = row.replace('<','').split('>')[0:-1]
        df.loc[i,'Len_tags'] = len(tags)
        for j,element in enumerate(tags) :
            df.loc[i,'Tag_'+str(j)]= element
            if element in dict_tags.keys():
                dict_tags[element]['List_id'].append(df.Id[i])
                dict_tags[element]['List_tags'].append(len(tags)-1)
            else :
                dict_tags[element]=dict()
                dict_tags[element]['List_id'] = [df.Id[i]]
                dict_tags[element]['List_tags'] = [len(tags)-1]
    df_tags = pd.DataFrame(index=dict_tags.keys(),columns=['Number_tags','Mean_tags'])
    for tags in df_tags.index :
        df_tags.loc[tags,['Number_tags']] = len(dict_tags[tags]['List_id'])
    df_tags = df_tags.sort_values(['Number_tags'],ascending=False)
    list_best_tags = df_tags[0:30].index
    df_best_tags = df_tags.loc[list_best_tags]
    df_30_tags = pd.concat(df[df['Id'].isin(
                    dict_tags[tags]['List_id'])] for tags in list_best_tags).drop_duplicates().drop(
        'Tags',axis=1).reset_index(drop=True)
    cleaning_df = df_30_tags[['Tag_0','Tag_1','Tag_2','Tag_3','Tag_4']].isin(list_best_tags)
    for j in range(5):
        df_30_tags['Tag_'+str(j)][cleaning_df['Tag_'+str(j)]!= True] = np.nan
    df_30_tags['New_len_tags'] = cleaning_df.sum(axis=1)
    df_30_tags['Tags'] =  df_30_tags.apply(lambda x : Tags_to_list(x),axis=1)
    df_30_tags.drop(['Tag_0','Tag_1','Tag_2','Tag_3','Tag_4'],axis=1,inplace=True)
    return df_best_tags,df_30_tags
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

def transform_text_bow_lem_spacy_fct(desc_text) :
    text = text_extract(desc_text)
    word_tokens = list(sent_to_words(text))

    lem_w = lemmatization_spacy(word_tokens)
    sw = remove_stopwords(lem_w)
    transf_desc_text = ' '.join(lem_w[0])
    return transf_desc_text

df_study['Text_Title'] = df_study['Title_bow_lem'].apply(transform_bow_lem_spacy_fct)+' '+df_study['Text_bow_lem'].apply(transform_text_bow_lem_spacy_fct)
