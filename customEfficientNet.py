from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class EfficientNetCustom(nn.Module):
  def __init__(self, model_name, input_channels, num_classes,
               load_pretrained_weights=True, train_only_last_layer=False):
    super(EfficientNetCustom, self).__init__()
    self.model_name = model_name
    self.input_channels = input_channels
    self.num_classes = num_classes
    self.load_pretrained_weights = load_pretrained_weights
    self.train_only_last_layer = train_only_last_layer
    if self.load_pretrained_weights:
      self.features = EfficientNet.from_pretrained(self.model_name,
                                               in_channels=self.input_channels)
    else:
      self.features = EfficientNet.from_name(self.model_name,
                                         input_channels=self.input_channels)
    if self.train_only_last_layer:
      for param in self.features.parameters():
        param.requires_grad = False
    in_ftrs = self.features._fc.in_features
    # self.features._fc = nn.Linear(in_ftrs, self.num_classes)
    self.classifier_layer = nn.Sequential(
            nn.Linear(in_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512 , self.num_classes)
            # nn.Linear(256 , self.num_classes)
        )

  def forward(self, inputs):
    x = self.features.extract_features(inputs)
    x = self.features._avg_pooling(x)
    x = x.flatten(start_dim=1)
    x = self.features._dropout(x)
    x = self.classifier_layer(x)
    return x