import pickle
from flask import Flask,jsonify,render_template,request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# import ridge regressor model and standard scalar pickle file
ridge_model = pickle.load(open('LinR_project/Models/ridge.pkl','rb'))
standard_scalar = pickle.load(open('LinR_project/Models/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_stand_scaler = standard_scalar.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_stand_scaler)

        return render_template('home.html',result = result)
    else:
        return render_template('home.html')



if __name__ == "__main__":
    app.run()