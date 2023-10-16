#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from flask import Flask, request, json, jsonify
from random import randint, uniform
from sklearn.metrics import confusion_matrix

import json
import os
import pickle

import predictions as pr

app = Flask(__name__)

df_original = pd.read_csv('data/aps_failure_training_set.csv', dtype = 'str')
df_original = df_original.replace(r'na', 0, regex=True)
df_original = df_original.drop(df_original.iloc[:, 5:-1],axis = 1)
df_original['aa_000'] = df_original['aa_000'].astype(int)
df_original['ab_000'] = df_original['ab_000'].astype(int)
df_original['ac_000'] = df_original['ac_000'].astype(int)
df_original['ad_000'] = df_original['ad_000'].astype(int)
df_original['eg_000'] = df_original['eg_000'].astype(int)

truck_fleet= ['AB100CD','AB200CD','AB300CD','AB400CD','AB500CD','AB600CD','AB700CD','AB800CD','AB900CD','AB110CD','AB120CD','AB130CD','AB140CD','AB150CD']

@app.route("/")
def hello():
    return jsonify(message="Hello! I'm another Failures Classifier")


@app.route('/classify', methods=['GET'])
def classify():
    """
        Classify the failures based on truck patent.
    """
    # Get the url params.
    truck_patent = request.args.get('truck_patent')
    
    sample = []
    
    if truck_patent in truck_fleet:
        aa_000 = int(randint(0,df_original['aa_000'].max()))
        ab_000 = int(randint(0,df_original['ab_000'].max()))
        ac_000 = int(randint(0,df_original['ac_000'].max()))
        ad_000 = int(randint(0,df_original['ad_000'].max()))
        eg_000 = int(randint(0,df_original['eg_000'].max()))
        sample = [aa_000, ab_000, ac_000, ad_000, eg_000]   
        
    # Validations 101
    if not sample:
        return jsonify(error='Patente no encontrada. Ingrese otra'), 404
    
    try:
        result = pr.predict(sample)
        return jsonify(truck_patent,result)
    except Exception as e:
        return jsonify(error=str(e))


if __name__ == '__main__':    
    app.run(host='0.0.0.0', debug=True) 
