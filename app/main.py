from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import io
import requests
import json
import onnxruntime as ort
import numpy as np

app = FastAPI()

# Load the pre-trained model from PyTorch Hub
model = '../models/resnet101-v2-7.onnx'
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(model, providers=providers)


# Load the class labels for the pre-trained model
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(requests.get(LABELS_URL).text)

# Preprocess the input image
def preprocess_image(image: Image, height, width) -> Image:
    resized_image = image.resize((height, width))  # Assuming input size of the model is 224x224
    normalized_image = np.array(resized_image) / 255.0  # Normalize pixel values between 0 and 1
    preprocessed_image = np.transpose(normalized_image, (2, 0, 1))  # Reshape image to (C, H, W) format
    input_tensor = preprocessed_image[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor


def infer(image: Image):

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name

    input_height, input_width = input_shape[2:]
    input_tensor = preprocess_image(image, input_height, input_width)
    
    outputs = session.run([output_name], {input_name: input_tensor})

    # Postprocess the outputs
    class_probabilities = outputs[0]
    predicted_class_index = np.argmax(class_probabilities)
    probabilities = class_probabilities[0][predicted_class_index]
    # Get the predicted class label
    return {'classes': labels[predicted_class_index], 'probabilities': float(probabilities)}



@app.post('/inference')
async def inference(file: UploadFile = File(...)):
    
    # Read the image file
    image = Image.open(io.BytesIO(await file.read()))

    # return result inferencing 
    return infer(image)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):

    # Read the image file
    image = Image.open(io.BytesIO(await file.read()))

    output = infer(image)

    # Business logic
    if output['classes'] == 'Border Collie':
        response = {'classification': 'iO'}
    else:
        response = {'classification': 'NiO'}

    # Return response
    return response

if __name__ == '__main__':
    app.run(debug=True)
