# import library
import torch
from model import Generator
from PIL import Image
import numpy as np 

# set parameter
img_size = 512
batch_size = 1
onnx_model_path = 'weights.onnx'
torch_model_path = 'weights.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = Network()
model.load_state_dict(torch.load(torch_model_path, map_location='cpu'))
model.eval()

# create input and check model
sample_input = torch.rand(batch_size, img_size, img_size, 3) 
y = model(sample_input)

#convert model onnx
torch.onnx.export(
    model,
    sample_input, 
    onnx_model_path,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)
