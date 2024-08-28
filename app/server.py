# from fastapi import FastAPI, UploadFile, File
# import joblib
# import numpy as np
# import uvicorn
# from prediction import *
#
# #model = joblib.load('disease_classifier.joblib')
#
# app = FastAPI()
#
# @app.get('/index')
# def classifier_model():
#     return "Dr.Derm Disease Classifier"
#
# @app.post('/api/predict')
# async def predict_image(file: UploadFile = File(...)):
#     # You would add the logic here to handle the image file and make a prediction
#     image = read_image(await file)
#     return {"message": "Prediction logic to be added here"}
#
# if __name__ == "__main__":
#     uvicorn.run(app, port=8080, host='0.0.0.0')
#
from fastapi import FastAPI, UploadFile, File
import uvicorn
from prediction import read_image, preprocess, load_model, predict

# Load the model at startup
model = load_model('disease_classifier.joblib')

app = FastAPI()

@app.get('/index')
def classifier_model():
    return "Dr.Derm Disease Classifier"

@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    # Read the uploaded image
    image = read_image(await file.read())

    # Preprocess the image
    preprocessed_image = preprocess(image)

    # Make a prediction using the model
    prediction = predict(preprocessed_image, model)

    # Return the prediction result
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
