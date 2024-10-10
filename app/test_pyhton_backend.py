import requests
ENDPOINT = 'http://localhost:8080/test'

def test_can_call_endpoint():
    response = requests.get(ENDPOINT)
    #assert response.json()["status_code"] ==200
    assert response.status_code ==200



# URL constants
INDEX_ENDPOINT = 'http://localhost:8080/index'
PREDICT_ENDPOINT = 'http://localhost:8080/api/predict'


# Test for /index endpoint
def test_can_call_index_endpoint():
    response = requests.get(INDEX_ENDPOINT)
    assert response.status_code == 200
    assert response.json()["message"] == "Dr.Derm Disease Classifier"


# Test for /api/predict endpoint
def test_can_call_predict_endpoint():
    # Open a sample image file to send as multipart data
    # Use a real image file path or mock it in your test environment
    image_path = r'C:\Users\Asus\Desktop\Dr.derm\deployment_tuto\app\melanoma 2.jpg'

    with open(image_path, "rb") as image_file:
        files = {"file": image_file}

        response = requests.post(PREDICT_ENDPOINT, files=files)

        # Check if the request was successful
        assert response.status_code == 200

        # You can further check the response structure
        prediction_data = response.json()
        assert "message" in prediction_data or "prediction" in prediction_data["data"]

