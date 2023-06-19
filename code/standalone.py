from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import requests
import json
from timeit import default_timer



# Load the pre-trained model from PyTorch Hub
start_load = default_timer()
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
duration_load = default_timer() - start_load
print("Time (load): {}".format(duration_load))

model.eval()

image_file = Image.open('images/dog-and-cat.jpeg')

# Set up the image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the class labels for the pre-trained model
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(requests.get(LABELS_URL).text)


def predict(image: Image):
    start_predict = default_timer()
    # Read the image file

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Perform the inference
    with torch.no_grad():
        start_inference = default_timer()
        output = model(input_batch)
        duration_inference = default_timer() - start_inference

    # Get the predicted class label
    _, predicted_idx = torch.max(output, 1)
    predicted_label = labels[predicted_idx.item()]

    # Return the predicted label as a response
    duration_predict = default_timer() - start_predict
    return {'prediction': predicted_label, 'time': {'inference': duration_inference, 'predict': duration_predict}}

if __name__ == '__main__':
    start_main = default_timer()
    output = predict(image_file)
    duration_main = default_timer() - start_main
    print("Time (main: {}".format(duration_main))

    predictions = output['prediction']
    time_predict = output['time']['predict']
    time_inference = output['time']['inference']
    print('Prediction: {}, Time (predict): {}, Time (inference): {}'.format(predictions, time_predict, time_inference))
