#
# from inference_sdk import InferenceHTTPClient
#
# # initialize the client
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="Re2yQfbDiRNpNpjwoReH"
# )
#
# # infer on a local image
# result = CLIENT.infer("C:/Users/Asus/Desktop/Dr.derm/deployment_tuto/app/melanoma 2.jpg", model_id="beauty_face/3")
# print(type(result), result)
# print(result["predictions"])
#
#
# import cv2
# from google.colab.patches import cv2_imshow
#
# # Load your image (change the path to the actual image location)
# image_path = "C:/Users/Asus/Desktop/Dr.derm/deployment_tuto/app/melanoma 2.jpg"
# image = cv2.imread(image_path)
#
# # Prediction results (from your data)
# predictions =  result["predictions"]
#
# # Draw rectangles for each prediction
# for pred in predictions:
#     x = int(pred['x'])
#     y = int(pred['y'])
#     width = int(pred['width'])
#     height = int(pred['height'])
#
#     # Calculate the top-left corner
#     top_left = (x - width // 2, y - height // 2)
#     bottom_right = (x + width // 2, y + height // 2)
#
#     # Draw rectangle
#     cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
#
#     # Add label
#     label = f"{pred['class']} ({pred['confidence']:.2f})"
#     cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
# # Display the image using cv2_imshow (since cv2.imshow is disabled in Colab)
# cv2_imshow(image)
#
# # Optionally save the marked image
# cv2.imwrite('marked_image.jpg', image)



from inference_sdk import InferenceHTTPClient
import cv2

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Re2yQfbDiRNpNpjwoReH"
)

# infer on a local image
image_path = "C:/Users/Asus/Desktop/Dr.derm/deployment_tuto/app/melanoma 2.jpg"
result = CLIENT.infer(image_path, model_id="beauty_face/3")
print(type(result), result)
print(result["predictions"])

# Load your image
image = cv2.imread(image_path)

# Prediction results (from your data)
predictions = result["predictions"]

# Draw rectangles for each prediction
for pred in predictions:
    x = int(pred['x'])
    y = int(pred['y'])
    width = int(pred['width'])
    height = int(pred['height'])

    # Calculate the top-left corner
    top_left = (x - width // 2, y - height // 2)
    bottom_right = (x + width // 2, y + height // 2)

    # Draw rectangle
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Add label
    label = f"{pred['class']} ({pred['confidence']:.2f})"
    cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the image in a window (for local machine)
cv2.imshow('Predictions', image)

# Wait for a key press and then close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the marked image
cv2.imwrite('marked_image.jpg', image)
