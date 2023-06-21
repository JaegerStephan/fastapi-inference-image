import requests
from timeit import default_timer

# Specify the URL of the FastAPI server endpoint
url = 'http://localhost:56414/inference'

# Read the image file
image_files = ['images/dog-and-cat.jpeg', 'images/cat.jpeg']
files = [('files', open(file, 'rb')) for file in image_files]

# Send a POST request to the FastAPI server with the image file
start = default_timer()
response = requests.post(url, files=files, timeout=1)
print(default_timer() - start)
# Process the response
if response.status_code == 200:
    print(response.json())
    # predictions = response.json()['labels']
    # print('Prediction: {}'.format(predictions))
else:
    print('Error:', response.text)
