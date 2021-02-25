import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
import numpy as np
import os
import cv2

def index_to_label(idx, test_dir):
  classes = sorted(os.listdir(test_dir))
  temp = classes[idx]
  return temp

def preprocess_img(img):
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (image_size, image_size))
  img = np.asarray(img, dtype=np.float32)
  rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
  img = (img.astype(np.float32))/255.0
  img = (img-rgb_mean)/(rgb_std)
  img = np.transpose(img, (2,0,1))
  image= np.expand_dims(img, 0)
  return image

def predict(model_path, preprocessed_img):
  import onnx
  import onnxruntime as ort
  onnx_model = ort.InferenceSession(model_path)
  pred = onnx_model.run(None, {'input.1': preprocessed_img})
  pred = np.argmax(pred)
  return pred

model_name = 'efficientnet-b2'
global image_size 
image_size = EfficientNet.get_image_size(model_name)
model_path = '/content/drive/MyDrive/Inspiring/models/pollen-eff-b2.onnx'
test_dir = '/content/drive/MyDrive/Inspiring/datasets_test'

res = []
for root_dir, subdir, files in os.walk(test_dir):
  for img in files:
    img_path = os.path.join(root_dir, img)
    pp_img = preprocess_img(img_path)
    pred = predict(model_path, pp_img)
    res.append(index_to_label(pred, test_dir))
from collections import Counter
counts = Counter(res)
print(counts)