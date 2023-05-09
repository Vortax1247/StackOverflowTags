import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_text.pkl', 'rb'))


   
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model_text.predict(text_processing(request))
    tags = multibinazieer.inverse_transform(prediction)
    return jsonify(tags)

if __name__ == "__main__":
    app.run(debug=True)
