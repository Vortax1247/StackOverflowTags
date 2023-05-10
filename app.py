import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from text_preprocess import text_preprocessing

app = Flask(__name__)
model = pickle.load(open('model_text.pkl', 'rb'))
multibinazier = pickle.load(open('binarizer.pkl','rb'))

   
@app.route('/results',methods=['POST'])
def results():
    data = pd.DataFrame(request.get_json())
    prediction = model.predict(text_preprocessing(data))
    tags = multibinazier.inverse_transform(prediction)
    return jsonify(tags)


if __name__ == "__main__":
    app.run(debug=True)
