import multiprocessing
from efficientnet_pytorch import EfficientNet

root_dir = '/content/drive/MyDrive/Inspiring/datasets_train'
model_name = 'efficientnet-b0'
image_size = EfficientNet.get_image_size(model_name)
rgb_mean = [0.0176, 0.0169, 0.0160]
rgb_std = [0.1016, 0.0977, 0.0905]
num_workers = multiprocessing.cpu_count()
train_batch_size = 128
valid_batch_size = 64
num_epochs = 20
train_split = 0.8
learning_rate = 0.0001
load_pretrained_weights = True
train_only_last_layer = False