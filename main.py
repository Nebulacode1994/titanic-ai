from fastapi import FastAPI
import pickle
import numpy as np
import joblib # needed to load the model
import os # needed for BASE_DIR and the path logic

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'titanic_model.pkl')
model = joblib.load(model_path)


@app.get("/")
def home():
    return {"status":"Server is Online", "message": "Go go /docs for UI"}

@app.get("/predict")
def predict(pclass: int, sex_male: bool, age: float, sibsp: int, parch: int):
    
    gender = 1 if sex_male else 0
    
    features = np.array([[pclass, gender, age, sibsp, parch]])
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    result = "Survived" if prediction[0] == 1 else "did not survive"
    
    return{ "prediction": result, "confidence_score": round(float(np.max(probablity)), 2),"input_received": {"class": pclass,"age":age,"is_male": sex_male}}
