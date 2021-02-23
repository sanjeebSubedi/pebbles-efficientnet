import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
import numpy as np

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
            nn.Linear(512, 256),
            nn.Linear(256 , 9)
        )

  def forward(self, inputs):
    x = self.features.extract_features(inputs)
    x = self.features._avg_pooling(x)
    x = x.flatten(start_dim=1)
    x = self.features._dropout(x)
    x = self.classifier_layer(x)
    return x


def index_to_label(idx):
  indices = {'ACE': 0, 'BET': 1, 'CORSTER': 2, 'CUP': 3, 'FRA': 4, 'MOR': 5, 'PLA': 6, 'POP': 7, 'SAL': 8}
  indices = {y:x for x,y in indices.items()}
  temp = indices.get(idx)
  return temp

def preprocess_img(img):
  if isinstance(img, str):
    image = Image.open(img).convert('RGB')
  elif isinstance(img, np.ndarray):
    image = Image.fromarray(img).convert('RGB')
  elif isinstance(img, torch.Tensor):
    image = Image.fromarray(img.numpy()).convert('RGB')
  else:
    raise Exception(f'type of `img` should be any of tensor, numpy array or a path to an image, not {type(img)}.')
  
  tfms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize([0.0176, 0.0169, 0.0160],
                                                       [0.1016, 0.0977, 0.0905])])
  image = tfms(image)
  image = image.unsqueeze(0)
  return image
  
def predict(model, preprocessed_img):
  model.eval()
  pred = model(preprocessed_img)
  pred = np.argmax(pred.detach().numpy())
  res = index_to_label(pred)
  return res

model_name = 'efficientnet-b0'
model_path = '/content/drive/MyDrive/Inspiring/models/pollen-eff-b2.pt'
img_path = '/content/drive/MyDrive/Inspiring/datasets_train/POP/20191121T003200362-147-5.png'

loaded_model = InferenceModel(model_name)
loaded_model.load_state_dict(torch.load(model_path, map_location='cpu'))

pp_img = preprocess_img(img_path)
res = predict(loaded_model, pp_img)
print(res)