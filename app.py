from tensorflow import keras
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
import os 

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'models')

app=Flask(__name__)

@app.route('/')
def welcome():
    return ""

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predciting the price of a given appartment.
    input:
        json file containing the following keys :
            address: which contains a dictionary with two keys: lat and lon.
            dataValues: which contains a dictionary with the following keys: 
                builyear, usableArea, primærrom, floorNumber, bedrooms, propertyNumber: str
                heis: str, value Heis. Should not exist in the dictionary if there is no heis
                takterrasse: str, none, Should not exist in the dictionary if there is no terrasse or balkony
                garasjeplass: str, value Garasjeplass. Should not exist in the dictionary if there is no heis
        i.e.: {"address":{"lat":59.913486,"lon":10.723724},
                "dataValues":{"heis":"Heis","garasjeplass":"Garasjeplass","propertyNumber":"H0102","floorNumber":"1","bedrooms":"1","primærrom":"47","usableArea":"42","builtYear":"1998"}}
    Response:
        dict['prediction'] contains the prediction: float
    '''
    # Load the model components
    model1 = joblib.load(os.path.join(model_path, 'model_EL.pkl'))
    model_pipeline = joblib.load(os.path.join(model_path, 'model_pipeline.pkl'))
    model2= keras.models.load_model(os.path.join(model_path))
    appartment = {}
    data=request.get_json()
    if('garasjeplass' in data['dataValues'].keys()):
        appartment['F_GarasjeP-plass']=1
    else:
        appartment['F_GarasjeP-plass']=0
    if('takterrasse' in data['dataValues'].keys()):
        appartment['F_BalkongTerrasse']=1
    else:
        appartment['F_BalkongTerrasse']=0
    if('heis' in data['dataValues'].keys()):
        appartment['F_Heis']=1
    else:
        appartment['F_Heis']=0
    appartment['buildyear']=int(data['dataValues']['builtYear'])
    appartment['bedrooms']=int(data['dataValues']['bedrooms'])
    appartment['floor']=int(data['dataValues']['floorNumber'])
    appartment['lng']=float(data['address']['lon'])
    appartment['lat']=float(data['address']['lat'])
    appartment['PROM']=int(data['dataValues']['primærrom'])
    appartment['BRA']=int(data['dataValues']['usableArea'])
    appartment['apartmentnumber']=data['dataValues']['propertyNumber']
    appartment['adcreated']=str(datetime.now())
    appartment = pd.DataFrame(appartment, index=[1])
    prediction=model_pipeline.predict(appartment,model1=model1,model2=model2)
    return jsonify({'prediction_EnsembleLearning': prediction[0][0],'prediction_Keras': prediction[1][0],'prediction': prediction[2][0]}), 200

if __name__=='__main__':
    app.run(host='0.0.0.0')