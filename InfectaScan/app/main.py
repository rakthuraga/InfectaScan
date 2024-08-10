from flask import Flask, request, Response, jsonify
from flask_cors import CORS

# libraries we need
import torch
import numpy as np
# import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# from torchvision.transforms import v2
import torch.nn as nn
# import torch.nn.functional as F

# from tqdm.notebook import tqdm
import io, base64
from PIL import Image

app = Flask(__name__)
CORS(app)

#An essential feauture of pytorch is its ability to utlize GPUs and TPUs
#This portion of the code chooses the device that while be used,
#if there is a GPU that uses the CUDA toolkit, then it will utlize it
#if not it will use the CPU for calculations
#Note many times, GPUs are much faster option, and allow for more realistic training times

if torch.cuda.is_available(): # Checks if CUDA is availiable, loads the device for computation to the GPU
    device = torch.device('cuda:0')
    print('Running on GPU')
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('Running on CPU')

#This is a class module for to create a CNN, not the Module class in pytorch
#Is the base class for all models in pytorch, this contains the inner working of a module

#Functions ->
# The def __init__(self) is a constructor, where you outline the different layers and aspects of your custom class
# def forward is the function for forward propogationm you give it an input X and it outputs tensore

#Layers ->
#In pytorch a nn.Conv2d layer is a convolution 2d layer, the arguments are as follows
#nn.Conv2d(Number of Input features maps, Number of features maps, Kernel Size, Stride Size, Padding Size )
#nn.BatchNorm2d is a batch normalization layer that takes in a 2d tensor the argument is the number of input feature maps
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First convolutional layer
        # Here we're defining a standard layer with Convolution, BatchNorm, and dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)  # b x 3 x 32 x 32 -> b x 32 x 16 x 16
        self.batchnorm1 = nn.BatchNorm2d(32)                               # (channel x height x width), b is batch size
        self.relu1 = nn.ReLU()  # Using ReLU activation function
        self.dropout1 = nn.Dropout(0.1)  # Adding dropout to prevent overfitting

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)  # b x 32 x 16 x 16 -> b x 64 x 8 x 8
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # Adding a pooling layer to reduce spatial dimensions, b x 64 x 8 x 8 -> b x 64 x 4 x 4
        self.dropout2 = nn.Dropout(0.05)

        # Third convolutional layer
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # b x 64 x 4 x 4 -> b x 64 x 4 x 4
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)  # b x 64 x 4 x 4 -> b x 64 x 4 x 4
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.05)
        self.flatten = nn.Flatten()  # b x 64 x 4 x 4 -> b x (64 * 4 * 4)

        # Fully connected layer - classifying the features into 10 classes
        self.fc = nn.Linear(64 * 4 * 4, 128)  # 64 from the last conv layer, 10 for the number of classes, b x (64 * 4 * 4) -> b x 128
        # self.bachnorm4 = nn.BatchNorm1d(128)
        self.relu4 =  nn.ReLU()
        self.dropout4 = nn.Dropout(0.05)

        # self.fc0 = nn.Linear(128, 64)
        # self.relu5 =  nn.ReLU()
        self.fc1 = nn.Linear(128, 33)  # b x 128 -> b x 10

    def forward(self, x):
        # Describing the forward pass through the network
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.flatten(x)  # Flattening the output of the conv layers for the fully connected layer
        x = self.fc(x)
        # x = self.bachnorm4(x)
        x = self.relu4(x)
        # x = self.dropout4(x)

        # x = self.fc0(x)
        # x = self.relu5(x)
        x = self.fc1(x)
        return x  # The softmax (or another activation) can be implicitly applied by the loss function

# We are creating an instance of our CNN model, after which we load to model to
# the device either GPU or CPU
model = CNN()
model.load_state_dict(torch.load("./model/model.pth"))
model.to(device)

read_in_array_sources = np.loadtxt("database_sources.txt", dtype="str", delimiter=":")
classes = read_in_array_sources[:,0]

transform2 = transforms.Compose([
    transforms.Resize(256),
    # transforms.RandomRotation(30),
    transforms.CenterCrop(64),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5], 
        [0.5, 0.5, 0.5])])

# Health check route
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

# Predict route
# @app.route("/predict", methods=["POST"])
# def predict():
#     print("/predict request")
#     req_json = request.get_json()
#     json_instances = req_json["instances"]
#     image = Image.open(io.BytesIO(base64.decodebytes(bytes(json_instances[0]["image"], "utf-8")))).convert('RGB')
#     image_transformed = transform2(image)
#     model.eval()
#     outputs = model(image_transformed.unsqueeze(0))
#     prediction = outputs.argmax(dim=1)
#     # print(preds_classes)

#     response = jsonify({
#     "prediction": classes[prediction]
# })
#     response.headers["Content-Type"] = "application/json"
#     return response

@app.route("/predict", methods=["POST"])
def predict():
    print("/predict request")

    # Check if the request contains the "image" field
    if "image" not in request.files:
        return Response("No 'image' field found in the form-data.", status=400)

    image_file = request.files["image"]

    if image_file.filename == "":
        return Response("No file selected.", status=400)

    # Process the image here
    image = Image.open(image_file.stream).convert('RGB')
    image_transformed = transform2(image)
    model.eval()
    outputs = model(image_transformed.unsqueeze(0))
    prediction = outputs.argmax(dim=1)

    response = jsonify({
        "prediction": classes[prediction]
    })
    response.headers["Content-Type"] = "application/json"
    return response



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
