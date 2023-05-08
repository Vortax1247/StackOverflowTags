
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from text_preprocess import *

import time
import logging

logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere


df_study     = pd.read_csv("tags_learning.csv").drop('Unnamed: 0',axis=1)
y  = pd.read_csv('tags_y_learning.csv').drop('Unnamed: 0',axis=1)

from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(data_T.Tags[0:-2])
Y = multilabel_binarizer.transform(data_T.Tags[0:-2])
pickle_out = open('binarizer.pkl', "wb")
pickle.dump(multilabel_binarizer, pickle_out)
pickle_out.close()
df_study['Text_Title'] = df_study['Title_bow_lem'].apply(transform_bow_lem_spacy_fct)+' '+df_study['Text_bow_lem'].apply(transform_text_bow_lem_spacy_fct)

batch_size = 10
sentences = df_study['Text_Title'].to_list()

features_USE = feature_USE_fct(sentences, batch_size)



model =  pickle.load("model_text.pkl",rb)
y_pred = model.predict(features_USE)
print_score(y,y_pred)
