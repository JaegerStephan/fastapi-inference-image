# Requirements
- Download onnx file for [ResNet](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet101-v2-7.onnx)


# Comparison of server client and standlone solution

## For a single image
- inferencing of the model takes always the same time (about 0.08 sec)
- the whole call of the prediction takes also always the same (about 0.14 sec)
- for the server client version, in addition to the standalone solution, there are about 0.01 sec for handling the request of an image of size (1862 x 1048 px)

## For two images
- inferencing of the model takes always the same time (about 0.23 sec)
- the whole call of the prediction takes also always the same (about 0.28 sec)


## Remote call (not localhost)
- if running client from another machine, the request for two images took about 0.6 sec while the inferencing itself still takes about 0.23 sec


# Error handling
- see http status codes on [mdn](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#client_error_responses)

# Podman

## Build
<pre>
podman build -t computer-vision-lever-variants .
</pre>