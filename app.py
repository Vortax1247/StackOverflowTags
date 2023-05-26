import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from text_preprocess import text_preprocessing

app = Flask(__name__)
model = pickle.load(open('model_text.pkl', 'rb'))
multibinazier = pickle.load(open('binarizer.pkl','rb'))


@app.route('/')
def index():
    return "<h1>Welcome the StackOverFlowTags-api!</h1>"

@app.route('/results',methods=['POST'])
def results():

    return jsonify(request.json)
