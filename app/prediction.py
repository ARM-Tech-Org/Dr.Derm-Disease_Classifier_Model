# from PIL import Image
# from io import BytesIO
# import numpy as np
# import tensorflow as tf
# import joblib
#
# # Define the input shape expected by the model
# input_shape = (244, 244)
#
# # Define the class labels
# class_labels = [
#     'Atopic Dermatitis',
#     'Eczema',
#     'Healthy',
#     'Melanoma',
#     'Psoriasis pictures Lichen Planus and related diseases',
#     'Seborrheic Keratoses and other Benign Tumors',
#     'Tinea Ringworm Candidiasis and other Fungal Infections',
#     'Warts Molluscum and other Viral Infections'
#
#
# def read_image(image_encoded):
#     """Read an image from the encoded file."""
#     return Image.open(BytesIO(image_encoded))
#
# # def preprocess(image: Image.Image) -> np.ndarray:
# #     """Preprocess the image to be suitable for model prediction."""
# #     image = image.resize(input_shape)
# #     img_array = np.array(image)
# #     img_array = np.expand_dims(img_array, axis=0)
# #     img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
# #     return img_array
#
# def preprocess(image: Image.Image) -> np.ndarray:
#     """Preprocess the image to be suitable for model prediction."""
#     image = image.resize(input_shape)  # Resize to 224x224
#     img_array = np.array(image)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
#     return img_array
#
#
# def load_model(model_path: str):
#     """Load the model from a .joblib file."""
#     return joblib.load(model_path)
#
# def predict(image: np.ndarray, model) -> dict:
#     """Make a prediction using the pre-trained model."""
#     # Predict the image
#     predictions = model.predict(image)
#
#     # Sort predictions to get the indices of the top two classes
#     top_two_indices = np.argsort(predictions[0])[-2:][::-1]
#
#     # Get the labels and probabilities of the top two classes
#     predicted_class_1 = top_two_indices[0]
#     predicted_label_1 = class_labels[predicted_class_1]
#     predicted_probability_1 = predictions[0][predicted_class_1]
#
#     predicted_class_2 = top_two_indices[1]
#     predicted_label_2 = class_labels[predicted_class_2]
#     predicted_probability_2 = predictions[0][predicted_class_2]
#
#     # Return the results as a dictionary
#     return {
#         "predicted_class_1": predicted_label_1,
#         "predicted_probability_1": float(predicted_probability_1),
#         "predicted_class_2": predicted_label_2,
#         "predicted_probability_2": float(predicted_probability_2)
#     }

from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Correctly importing preprocess_input
import joblib

# Define the input shape expected by the model
input_shape = (244, 244)  # Standard MobileNet input size

# Define the class labels
class_labels = [
    'Atopic Dermatitis',
    'Eczema',
    'Healthy',
    'Melanoma',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Warts Molluscum and other Viral Infections'
]

def read_image(image_encoded):
    """Read an image from the encoded file."""
    return Image.open(BytesIO(image_encoded))

def preprocess(image: Image.Image) -> np.ndarray:
    """Preprocess the image to be suitable for model prediction."""
    image = image.resize(input_shape)  # Resize to 224x224
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Proper usage of preprocess_input
    return img_array

def load_model(model_path: str):
    """Load the model from a .joblib file."""
    return joblib.load(model_path)  # Correct usage of joblib.load

def predict(image: np.ndarray, model) -> dict:
    """Make a prediction using the pre-trained model."""
    # Predict the image
    predictions = model.predict(image)

    # Sort predictions to get the indices of the top two classes
    top_two_indices = np.argsort(predictions[0])[-2:][::-1]

    # Get the labels and probabilities of the top two classes
    predicted_class_1 = top_two_indices[0]
    predicted_label_1 = class_labels[predicted_class_1]
    predicted_probability_1 = predictions[0][predicted_class_1]

    predicted_class_2 = top_two_indices[1]
    predicted_label_2 = class_labels[predicted_class_2]
    predicted_probability_2 = predictions[0][predicted_class_2]

    # Return the results as a dictionary
    return {
        "predicted_class_1": predicted_label_1,
        "predicted_probability_1": float(predicted_probability_1),
        "predicted_class_2": predicted_label_2,
        "predicted_probability_2": float(predicted_probability_2)
    }
