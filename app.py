from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)

## load the 3 files we made 

model = joblib.load('titanic_model.pkl')
scaler = joblib.load('titanic_scaler.pkl')
le = joblib.load('gender_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    # get the test from the HTML form
    gender_text = request.form['sex']
    # encoder: text --> number
    gender_num = le.transform([gender_text])[0]
    
    # get other numbers from the form
    pclass = int(request.form['pclass'])
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    
    # combine into oe list
    features = [[pclass, gender_num , age, sibsp, parch]]
    
    # scaler: adjust the numbers
    features_scaled = scaler.transform(features)
    
    # model: predict
    prediction = model.predict(features_scaled)
    
    result = "Survived" if prediction[0] == 1 else "Did not survive"
    
    return render_template('index.html', prediction_text = result)

if __name__=="__main__":
    app.run(debug=True)
    
    