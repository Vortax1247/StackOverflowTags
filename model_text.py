
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from text_preprocess import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import pickle

df_study  = pd.read_json("tags_learning.json")
y  = pd.read_csv('tags_y_learning.csv').drop('Unnamed: 0',axis=1)



multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df_study.Tags)
y = multilabel_binarizer.transform(df_study.Tags)

X = text_preprocessing(df_study)
model = Pipeline([('tfidf', TfidfVectorizer(max_df=0.5, min_df=5, stop_words="english", max_features = 1000)),
          ('multi_svc', OneVsRestClassifier(LinearSVC()))])
X_train,X_test,y_train,y_test = train_test_split(X, y, train_size= 0.60, random_state=9000)

model.fit(X_train,y_train)
y_pred  = model.predict(X_test)
print_score(y_test,y_pred)

pickle_out = open('binarizer.pkl', "wb")
pickle.dump(multilabel_binarizer, pickle_out)
pickle_out.close()

pickle_out = open('model_text.pkl', "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
