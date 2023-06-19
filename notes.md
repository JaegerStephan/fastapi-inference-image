# Requirements
- Download onnx file for [ResNet](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet101-v2-7.onnx)


# Comparison of server client and standlone solution

- inferencing of the model takes always the same time (about 0.08 sec)
- the whole call of the prediction takes also always the same (about 0.14 sec)
- for the server client version, in addition to the standalone solution, there are about 0.01 sec for handling the request of an image of size (1862 x 1048 px)

# ToDo
- upload multiple images to request
- containerize and host with podman (containerd)