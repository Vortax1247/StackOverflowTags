import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from text_preprocess import text_preprocessing
import os

app = Flask(__name__)
model = pickle.load(open('model_text.pkl', 'rb'))
multibinazier = pickle.load(open('binarizer.pkl','rb'))


@app.route('/')
def index():
    return "<h1>Welcome the StackOverFlowTags-api!</h1>"

@app.route('/results',methods=['POST'])
def results():
    data = pd.DataFrame(request.get_json())
    prediction = model.predict(text_preprocessing(data))
    tags = multibinazier.inverse_transform(prediction)
    return jsonify(tags)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
