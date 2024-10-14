# import requests
# import uvicorn
# from tensorflow._api.v2.compat.v1 import app
#
# # Define the API endpoints
# INDEX_ENDPOINT = 'http://127.0.0.1:8080/index'
# TEST_ENDPOINT = 'http://127.0.0.1:8080/test'
# PREDICT_ENDPOINT = 'http://127.0.0.1:8080/api/predict'
#
#
# def test_can_call_index_endpoint():
#     """Test the /index endpoint"""
#     response = requests.get(INDEX_ENDPOINT)
#     assert response.status_code == 200
#
#     # Verify the JSON response structure
#     response_data = response.json()
#     assert response_data['status_code'] == 200
#     assert response_data['message'] == "Dr.Derm Disease Classifier"
#     assert response_data['data']['data'] == "Dr.Derm Disease Classifier"
#
#
# def test_can_call_test_endpoint():
#     """Test the /test endpoint"""
#     response = requests.get(TEST_ENDPOINT)
#     assert response.status_code == 200
#
#     # Verify the JSON response structure
#     response_data = response.json()
#     assert response_data['status_code'] == 200
#     assert response_data['message'] == "API works fine"
#     assert response_data['data']['example'] == "This is a sample response data"
#
#
# def test_can_predict():
#     """Test the /api/predict endpoint with an image upload"""
#     # Open an image file in binary mode for uploading
#     image_path = 'melanoma 2.jpg'  # Replace with your image path
#     with open(image_path, 'rb') as img_file:
#         files = {'file': img_file}
#         response = requests.post(PREDICT_ENDPOINT, files=files)
#
#     # Check if the request was successful
#     assert response.status_code == 200
#
#     # Verify the JSON response structure
#     response_data = response.json()
#     assert response_data['status_code'] == 200
#     assert response_data['message'] == "prediction successful"
#
#     # Check if 'prediction' is in the response data
#     assert 'prediction' in response_data['data']
#
# if __name__ == "__main__":
#     uvicorn.run(app, port=8080, host='0.0.0.0')



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
#=======================================================================================================================

# from fastapi import FastAPI, UploadFile, File
# import uvicorn
# from starlette.responses import JSONResponse
#
# from app.prediction import read_image, preprocess, load_model, predict
#
# # Load the model at startup
# #model = load_model('/code/app/disease_classifier.joblib')
#
# import joblib
# import os
#
# # Correct path for Docker environment
# model_path = os.path.join(os.path.dirname(__file__), "disease_classifier.joblib")
# model = joblib.load(model_path)
#
# app = FastAPI()
#
# @app.get('/index')
# def classifier_model():
#
#     response_data = {
#         "status_code": 200,
#         "message": "Dr.Derm Disease Classifier",
#         "data": {"name": "Dr.Derm Disease Classifier"}
#     }
#     return JSONResponse(content=response_data, status_code=200)
# @app.get('/test')
# def api_test():
#     response_data = {
#         "status_code": 200,
#         "message": "API works fine",
#         "data": {"example": "This is a sample response data"}
#     }
#     return JSONResponse(content=response_data, status_code=200)
#
# @app.post('/api/predict')
# async def predict_image(file: UploadFile = File(...)):
#     # Read the uploaded image
#     image = read_image(await file.read())
#
#     # Preprocess the image
#     preprocessed_image = preprocess(image)
#
#     # Make a prediction using the model
#     prediction = predict(preprocessed_image, model)
#
#     # Return the prediction result
#
#
#     response_data = {"status_code": 200,
#                      "message": "Prediction Success",
#                       "data": prediction}
#     return JSONResponse(content=response_data, status_code=200)
#
# @app.post('/api/beautyface')
# async def beautyface(file: UploadFile = File(...)):
#     # Read the uploaded image
#     image_data = await file.read()
#
#     # Convert the image bytes to a numpy array for processing
#     image = read_image(image_data)
#
#     # Preprocess the image
#     #preprocessed_image = preprocess(image)
#
#     # Make a prediction using the model (predictions contain bounding box data)
#     predictions = predict(image, bea
#
#     # Annotate the image with bounding boxes
#     annotated_image = draw_bounding_boxes(image, predictions)
#
#     # Convert the annotated image to a format that can be returned (e.g., PNG)
#     pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
#     buffer = BytesIO()
#     pil_image.save(buffer, format="PNG")
#     buffer.seek(0)
#
#     # Return the annotated image as a response
#     return StreamingResponse(buffer, media_type="image/png")
#
#
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, port=8080, host='0.0.0.0')
#

#=========================================================================================================================

from fastapi import FastAPI, UploadFile, File
import uvicorn
from starlette.responses import JSONResponse, StreamingResponse
from app.prediction import read_image, preprocess, load_model, predict
import joblib
import os
from inference_sdk import InferenceHTTPClient
import cv2
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load the disease classification model at startup
model_path = os.path.join(os.path.dirname(__file__), "disease_classifier.joblib")
model = joblib.load(model_path)

# Initialize the inference client for object detection (beauty face detection)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Re2yQfbDiRNpNpjwoReH"
)

@app.get('/index')
def classifier_model():
    response_data = {
        "status_code": 200,
        "message": "Dr.Derm Disease Classifier",
        "data": {"name": "Dr.Derm Disease Classifier"}
    }
    return JSONResponse(content=response_data, status_code=200)

@app.get('/test')
def api_test():
    response_data = {
        "status_code": 200,
        "message": "API works fine",
        "data": {"example": "This is a sample response data"}
    }
    return JSONResponse(content=response_data, status_code=200)

# Disease classification endpoint
@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    # Read the uploaded image
    image = read_image(await file.read())

    # Preprocess the image
    preprocessed_image = preprocess(image)

    # Make a prediction using the model
    prediction = predict(preprocessed_image, model)

    # Return the prediction result
    response_data = {
        "status_code": 200,
        "message": "Prediction Success",
        "data": prediction
    }
    return JSONResponse(content=response_data, status_code=200)

# Object detection (beauty face detection) endpoint
@app.post('/api/beautyface')
async def beautyface(file: UploadFile = File(...)):
    # Read the uploaded image
    image_data = await file.read()

    # Infer with the object detection model
    image = read_image(image_data)
    result = CLIENT.infer(file.filename, model_id="beauty_face/3")
    predictions = result["predictions"]

    # Draw bounding boxes on the image
    def draw_bounding_boxes(image, predictions):
        for pred in predictions:
            x = int(pred['x'])
            y = int(pred['y'])
            width = int(pred['width'])
            height = int(pred['height'])

            # Calculate the top-left and bottom-right corners
            top_left = (x - width // 2, y - height // 2)
            bottom_right = (x + width // 2, y + height // 2)

            # Draw rectangle and label
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            label = f"{pred['class']} ({pred['confidence']:.2f})"
            cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image

    annotated_image = draw_bounding_boxes(image, predictions)

    # Convert annotated image to PNG for response
    pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Return the annotated image as a response
    return StreamingResponse(buffer, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
