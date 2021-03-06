import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import math
import time
import pickle
import variables as vars
from customEfficientNet import EfficientNetCustom
from sam import SAM


def set_device():
    global device
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'Device set to {device}')
    return


def oversample_data(root_dir, dataset):
    '''Checks if sample weights is already saved on disk and creates sampler out of it. Creates
    sample weights if weights are not found on disk.'''
    if os.path.exists('/content/drive/MyDrive/Inspiring/pickles/class_weights.pkl'):
        with open('/content/drive/MyDrive/Inspiring/pickles/class_weights.pkl', 'rb') as f:
            sample_weights = pickle.load(f)
            f.close()
    else:
        class_weights = []
        for root, subdir, files in os.walk(root_dir):
            if len(files) > 0:
                class_weights.append(1/len(files))
        sample_weights = [0] * dataset_size
        for index, (data, label) in enumerate(dataset):
            class_weight = class_weights[label]
            sample_weights[index] = class_weight

        with open('/content/drive/MyDrive/Inspiring/pickles/class_weights.pkl', 'wb') as f:
            pickle.dump(sample_weights, f)
            f.close()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                    replacement=True)
    return sampler


def load_data(root_dir, transformations, train_split, num_workers, train_batch_size, valid_batch_size, is_test_set=False):
    dataset = ImageFolder(root_dir, transform=transformations)
    global indices
    indices = dataset.class_to_idx
    indices = {y: x for x, y in indices.items()}
    global num_classes, train_size, valid_size, dataset_size
    num_classes = len(dataset.classes)
    dataset_size = len(dataset)
    train_size = math.ceil(train_split*dataset_size)
    valid_size = dataset_size - train_size
    train_set, valid_set = torch.utils.data.random_split(
        dataset, [train_size, valid_size])
    sampler = oversample_data(root_dir, train_set)
    print(f'Oversampled data')
    train_loader = DataLoader(train_set, batch_size=train_batch_size, num_workers=num_workers,
                              sampler=sampler, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=valid_batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, valid_loader


def adam(model, lr):
  base_optimizer = optim.Adam
  optimizer = SAM(model.parameters(), base_optimizer, lr=lr)
  return optimizer


def criterion():
    return nn.CrossEntropyLoss()


def lr_scheduler(optimizer, factor, patience, verbose):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=factor,
                                                patience=patience, verbose=verbose)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def train_model(model, train_loader, optimizer, loss_func, num_epochs, lr_scheduler, valid_loader=None):
    model.to(device)
    model.train()
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        loss_after_epoch, accuracy_after_epoch = 0, 0
        num_labels = 0
        for index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            preds = model(images)
            loss = loss_func(preds, labels)

            # optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # optimizer.step()

            preds= model(images)
            loss_func(preds, labels).backward()
            optimizer.second_step(zero_grad=True)

            loss_after_epoch += loss
            accuracy_after_epoch += get_num_correct(preds, labels)
            num_labels += torch.numel(labels)
        loss_after_epoch /= 100
        print(f'Epoch: {epoch}/{num_epochs}  Acc: {(accuracy_after_epoch/num_labels):.5f}  Loss: {loss_after_epoch:.5f} Duration: {(time.time()-start_time):.2f}s',
              end='  ' if valid_loader is not None else '\n')
        if valid_loader is not None:
            if epoch == num_epochs:
                validate_model(model, valid_loader, loss_func,
                               lr_scheduler, plot_cm=True)
            else:
                validate_model(model, valid_loader, loss_func,
                               lr_scheduler, plot_cm=False)
    return


def validate_model(model, valid_loader, loss_func, lr_scheduler, plot_cm=False):
    model.to(device)
    model.eval()
    val_acc, val_loss = 0, 0
    true_labels, pred_labels = [], []
    num_labels = 0
    with torch.no_grad():
        for batch in valid_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            preds = model(images)
            val_loss += loss_func(preds, labels)
            val_acc += get_num_correct(preds, labels)
            num_labels += torch.numel(labels)
            if plot_cm:
                pred_labels += preds.argmax(dim=1)
                true_labels += labels

    val_acc /= num_labels
    print(f'Val_acc: {val_acc:.5f}  Val_loss: {(val_loss/100):.5f}')
    lr_scheduler.step(val_acc)
    if plot_cm:
        pred_labels = [x.item() for x in pred_labels]
        true_labels = [x.item() for x in true_labels]
        show_confusion_matrix(pred_labels, true_labels)
    return


def show_confusion_matrix(predicted_labels, true_labels):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(predicted_labels, true_labels)
    sns.heatmap(cm, annot=True, linewidths=.5, cmap='YlGnBu', annot_kws={"size": 10},
                fmt="g", xticklabels=[indices.get(x) for x in range(0, 9)],
                yticklabels=[indices.get(y) for y in range(0, 9)])
    return


def save_onnx(model):
    model.eval()
    model.to('cpu')
    model.features.set_swish(memory_efficient=False)
    input = torch.randn(2, 3, vars.image_size, vars.image_size)
    torch.onnx.export(model, input, vars.model_save_dir+'efficientnet-b2.onnx',
                      verbose=True)
    print('Model successfully saved in onnx format.')
    return


tfms = transforms.Compose([transforms.Resize(vars.image_size), transforms.ToTensor(),
                           transforms.Normalize(vars.rgb_mean, vars.rgb_std)])

set_device()
train_loader, valid_loader = load_data(vars.root_dir, transformations=tfms, num_workers=vars.num_workers,
                                       train_split=vars.train_split, train_batch_size=vars.train_batch_size,
                                       valid_batch_size=vars.valid_batch_size)
model = EfficientNetCustom(vars.model_name, input_channels=3, num_classes=num_classes,
                           load_pretrained_weights=vars.load_pretrained_weights,
                           train_only_last_layer=vars.train_only_last_layer)
optimizer = adam(model, vars.learning_rate)
loss_func = criterion()
scheduler = lr_scheduler(optimizer, factor=0.1, patience=3, verbose=2)
train_model(model, train_loader, optimizer=optimizer, loss_func=loss_func,
            num_epochs=vars.num_epochs, lr_scheduler=scheduler, valid_loader=valid_loader)
torch.save(model, '/content/drive/MyDrive/Inspiring/models/pollen-eff-b2.pt')
save_onnx(model)