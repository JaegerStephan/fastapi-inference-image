from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image
import io
import base64
import requests
import json
import onnxruntime as ort
import numpy as np
from timeit import default_timer
from pydantic import BaseModel

app = FastAPI()

# Load the pre-trained model from PyTorch Hub
model = '../models/resnet101-v2-7.onnx'
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(model, providers=providers)


# Load the class labels for the pre-trained model
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(requests.get(LABELS_URL).text)


class ModelInfo(BaseModel):
    name: str
    version: str
    description: str | None = None


class ImageClassificationInferenceResponse(BaseModel):
    classes: list[str | None] = None
    probabilities: list[float | None] = None

class ImageClassificationPredictionResponse(BaseModel):
    ok: bool
    classification: str
    probability: float

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


@app.get("/info")
async def info() -> ModelInfo:
    return ModelInfo(name="computer-vision-lever-variants",
                version="0.1",
                description="image classification for predicting the lever "
            "variant trained for Synact Evo brakes")

@app.post("/inference")
async def inference(files: list[UploadFile]):
    start_inference = default_timer()
    # Read the image files
    for file in files:
        image = Image.open(io.BytesIO(await file.read()))

    output = infer(image)

    print("time for inferencing: {}".format(default_timer() - start_inference))
    # return result inferencing
    return output

@app.post("/predict")
async def predict(files: list[UploadFile]) -> ImageClassificationPredictionResponse:
    start_predict = default_timer()
    # Read the image files
    for file in files:
        image = Image.open(io.BytesIO(await file.read()))
        
    output = infer(image)

    # Business logic
    if output['classes'] == 'Border Collie':
        # response = {'classification': 'iO'}
        response = ImageClassificationPredictionResponse(
            ok=True,
            classification="Border Collie",
            probability= 0.9)
        return response
    else:
        # response = {'classification': 'NiO'}
        # response = ImageClassificationPredictionResponse(
        #     ok=False,
        #     probability= 0.34)
        raise HTTPException(status_code=406, detail="It's not a Border Collie")


    print("time for predicting: {}".format(default_timer() - start_predict))
    # Return response
    # 

if __name__ == '__main__':
    app.run(debug=True)
