import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
import numpy as np
import os

class InferenceModel(nn.Module):
  def __init__(self, model_name):
    super(InferenceModel, self).__init__()
    self.model_name = model_name
    self.features = EfficientNet.from_name(self.model_name)
    in_ftrs = self.features._fc.in_features
    self.classifier_layer = nn.Sequential(
            nn.Linear(in_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 9)
            # nn.Linear(256 , 9)
        )

  def forward(self, inputs):
    x = self.features.extract_features(inputs)
    x = self.features._avg_pooling(x)
    x = x.flatten(start_dim=1)
    x = self.features._dropout(x)
    x = self.classifier_layer(x)
    return x


def index_to_label(idx, test_dir):
  classes = sorted(os.listdir(test_dir))
  temp = classes[idx]
  return temp

def preprocess_img(img):
  if isinstance(img, str):
    image = Image.open(img).convert('RGB')
  # elif isinstance(img, np.ndarray):
  #   image = Image.fromarray(img).convert('RGB')
  # elif isinstance(img, torch.Tensor):
  #   image = Image.fromarray(img.numpy()).convert('RGB')
  # else:
  #   raise Exception(f'type of `img` should be any of tensor, numpy array or a path to an image, not {type(img)}.')-
  tfms = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])
  image = tfms(image)
  image = image.unsqueeze(0)
  return image
  
def predict_pt(model_path, preprocessed_img):
  loaded_model = InferenceModel(model_name)
  loaded_model.load_state_dict(torch.load(model_path, map_location='cpu'))
  loaded_model.eval()
  pred = loaded_model(preprocessed_img)
  pred = np.argmax(pred.detach().numpy())
  return pred

def predict_onnx(model_path, preprocessed_img):
  import onnx
  import onnxruntime as ort
  onnx_model = ort.InferenceSession(model_path)
  pred = onnx_model.run(None, {'input.1': preprocessed_img.numpy()})
  pred = np.argmax(pred)
  return pred


model_name = 'efficientnet-b2'
global image_size 
image_size = EfficientNet.get_image_size(model_name)
model_path = '/content/drive/MyDrive/Inspiring/models/pollen-eff-b2.onnx'
model_onnx = True if '.onnx' in model_path else False
test_dir = '/content/drive/MyDrive/Inspiring/datasets_test'

res = []
for root_dir, subdir, files in os.walk(test_dir):
  for img in files:
    img_path = os.path.join(root_dir, img)
    pp_img = preprocess_img(img_path)
    if model_onnx:
      pred = predict_onnx(model_path, pp_img)
    else:
      pred = predict_pt(model_path, pp_img)
    res.append(index_to_label(pred, test_dir))
    
from collections import Counter
counts = Counter(res)
print(counts)