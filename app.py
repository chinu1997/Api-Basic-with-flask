from flask import Flask, request,jsonify
import numpy as np
import pickle as p
import pandas as pd
import json

app = Flask(__name__)

modelfile = 'models/regression_final.pickle'

model = p.load(open(modelfile, 'rb'))


@app.route('/')
def main():
    return ('Predict Boston House  API')


@app.route('/api/', methods=['POST'])
def Make_Prediction():
    try:
        j_data = request.get_json()
        try:
            prediction = np.array2string(model.predict(j_data))
            return jsonify(prediction)
        except:
            return "check your Model Input Again"
    except:
        return "Check your json data it should be like this format [[12,25,45,14,78,12,45]]"

