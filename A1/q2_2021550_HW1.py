# %% [markdown]
# ## Q2.1 Creating Dataset

# %% [markdown]
# #### Importing Libraries and basic setup

# %%
# importing the required libraries
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, Subset, SubsetRandomSampler
from torchvision import utils
from torchvision.transforms import v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# #### GPU Setup

# %%
# checking pytorch version
print(torch.__version__)

# %%
import torchvision
# checking if cuda is available
torch.cuda.is_available()

# %%
# setting the device to cuda is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# File handling
current_path = os.getcwd()
new_path = os.path.join(current_path, 'Cropped_final/Cropped_final')
os.chdir(new_path)
for i in os.listdir():
    print(i)

# %% [markdown]
# #### Dataset Class

# %%
class RussianWildLifeDataset(Dataset):
  def __init__(self, root_dir, transform):
    self.transform = transform
    self.root_dir = root_dir
    self.classes = os.listdir(self.root_dir)
    self.classes.sort()
    self.class_index_mapping = dict(zip(range(len(self.classes)), self.classes))
    self.__load_data__()
    self.num_classes = torch.unique(torch.tensor(self.labels)).size(0)
    
  def __load_data__(self):
    self.images_filenames = []
    self.labels = []
    for i in range(len(self.class_index_mapping)):
      # go to the class directory
      os.chdir(new_path)
      class_path = os.path.join(self.root_dir, self.class_index_mapping[i])
      for image in os.listdir(class_path):
        self.images_filenames.append(os.path.join(class_path, image))
        self.labels.append(i)

  def get_class_name(self, label):
    return self.class_index_mapping[label]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    image = im.open(self.images_filenames[idx])
    label = self.labels[idx]
    if (self.transform):
      image = self.transform(image)
    label_value = label
    sample = {'image': image, 'label': label}
    return sample

# %% [markdown]
# #### Transforms

# %%
composed_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# %% [markdown]
# #### Dataloader splitting

# %%
train_size = 0.7
val_size = 0.1
test_size = 0.2
batch_size = 16
run_num = 2

# %%
dataset = RussianWildLifeDataset(new_path, composed_transform)

# %%
class_wise_indices = [np.where(np.array(dataset.labels) == i)[0] for i in range(len(dataset.classes))]
class_distribution = [len(indices) for indices in class_wise_indices]
train_class_distribution = [int(train_size * dist) for dist in class_distribution]
val_class_distribution = [int(val_size * dist) for dist in class_distribution]
test_class_distribution = [int(test_size * dist) for dist in class_distribution]

# %%
train_indices = []
val_indices = []
test_indices = []
for i in range(len(dataset.classes)):
    np.random.shuffle(class_wise_indices[i])
    train_indices.extend(np.random.choice(class_wise_indices[i], train_class_distribution[i], replace=False))
    val_indices.extend(np.random.choice(np.setdiff1d(class_wise_indices[i], train_indices), val_class_distribution[i], replace=False))
    test_indices.extend(np.random.choice(np.setdiff1d(class_wise_indices[i], np.concatenate((train_indices, val_indices))), test_class_distribution[i], replace=False))

# %%
train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)
test_data = Subset(dataset, test_indices)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

# %%
train_size = len(train_data)
val_size = len(val_data)
test_size = len(test_data)
print(train_size, val_size, test_size)

# %% [markdown]
# #### Data Distribution Visualization

# %%
def plot_distribution(data, title):
  labels = [sample['label'] for sample in data]
  unique, counts = np.unique(labels, return_counts=True)
  plt.bar(unique, counts)
  plt.title(title)
  plt.show()

# %%
plot_distribution(train_data, 'Train set')

# %%
plot_distribution(val_data, 'Validation set')

# %% [markdown]
# #### WandB Setup

# %%
print(os.getcwd())
os.chdir("..")
os.chdir("..")
print(os.getcwd())

# %%
import wandb
wandb.login()

# %%
# force relogin
wandb.login(relogin=True)

# %%
model_config = dict(
    epochs=10,
    classes=10,
    kernels=[3, 3, 3],
    features=[32, 64, 128],
    batch_size=batch_size,
    optimizer="Adam",
    learning_rate=0.001,
    dataset="Russian WildLife Dataset",
    architecture="CNN", 
    weight_initialisation="Default")

# %%
wandb.init(
    project = "Computer_Vision_A1",
    config = model_config,
)

# %% [markdown]
# #### Data Visualization

# %%
def data_visualization(data):
    num_rows = 3
    num_cols = 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2*num_cols,2*num_rows))
    for i in range(num_rows):
        for j in range(num_cols):
            sample = train_data[np.random.randint(0, len(train_data))]
            image, label = sample['image'], sample['label']
            image = transforms.ToPILImage()((((sample['image']+1)/2) * 255).squeeze().to(torch.uint8))
            axes[i, j].imshow(image)
            axes[i, j].set_title(dataset.get_class_name(label))
            axes[i, j].axis('off')
    plt.show()

# %%
data_visualization(train_data)

# %%
data_visualization(val_data)

# %% [markdown]
# ## Q2.2 Training a CNN Model 

# %% [markdown]
# #### CNN Architecture and Model Definition

# %%
class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layer1 = nn.Sequential(
        # First Convolutional Layer
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=4, stride=4)
    )
    self.conv_layer2 = nn.Sequential(
        # Second Convolutional Layer
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.conv_layer3 = nn.Sequential(
        # Third Convolutional Layer
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.fully_connected = nn.Sequential(
        nn.Linear(128*14*14, 10)
    )
    self.Softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.conv_layer1(x)
    x = self.conv_layer2(x)
    x = self.conv_layer3(x)
    x = torch.flatten(x, 1)
    logits = self.fully_connected(x)
    return logits

  def softmax(self, x):
    x = self.Softmax(x)
    return x

# %%
model = ConvNet()
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# move the model to the gpu
model.to(device)

# %% [markdown]
# #### Logging Function

# %%
def loss_acc_calc(dataloader):
  model.eval()
  with torch.no_grad():
    correct, total, loss = 0, 0, 0.0
    for i, sample in enumerate(dataloader, 0):
      images = sample['image']
      labels = sample['label']
      images = images.to(device)
      labels = torch.tensor(labels).to(device)
      forwards = model.forward(images)
      logits = torch.nn.functional.softmax(forwards, dim=1)
      preds = torch.argmax(logits, dim=1)
      correct += torch.eq(preds, labels).sum().item()
      total += labels.size(0)
      loss_val = loss_fn(forwards, labels)
      loss += loss_val.item()
  return correct, total, loss

# %% [markdown]
# #### Model Training

# %%
wandb.watch(model)

# %%
def train_model(model, model_config, loss_fn, optimizer, train_dataloader, val_dataloader):
    running_loss = 0.0
    model.train()
    for epoch in range(model_config['epochs']):
        for i, sample in enumerate(train_dataloader, 0):
            inputs = sample['image']
            labels = sample['label']
            inputs = inputs.to(device)
            labels = torch.tensor(labels).to(device)
            # zero the gradients before backprop
            optimizer.zero_grad()
            # forward pass
            logits = model.forward(inputs)
            loss = loss_fn(logits, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            minibatch_loss = loss.item()
            running_loss += minibatch_loss
            # logging the minibatch after every batch, and running loss after every 100 minibatches
            wandb.log({'minibatch_loss': minibatch_loss})
            if (i%100==0):
                wandb.log({'running_loss': running_loss / 100})
                running_loss = 0.0
        # log the training accuracy and loss
        correct, total, loss = loss_acc_calc(train_dataloader)
        wandb.log({'train_loss': loss / len(train_dataloader), 'train_accuracy': (correct / total) * 100.0})
        # log the validation accuracy and loss
        correct, total, loss = loss_acc_calc(val_dataloader)
        wandb.log({'val_loss': loss / len(val_dataloader), 'val_accuracy': (correct / total) * 100.0})
    # save the model after training
    torch.save(model.state_dict(), f"A1_Q2_CNN_Experiment_{run_num}.pth")
    return model

# %%
# training the model 
model = train_model(model, model_config, loss_fn, optimizer, train_dataloader, val_dataloader)

# %% [markdown]
# #### Model Evaluation

# %%
def evaluation_metris_calc(dataloader):
    y_true = []
    y_pred = []
    for i, sample in enumerate(dataloader, 0):
        images = sample['image']
        labels = sample['label']
        images = images.to(device)
        labels = torch.tensor(labels).to(device)
        forwards = model.forward(images)
        logits = torch.nn.functional.softmax(forwards, dim=1)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred

# %%
# testing the model 
correct, total, loss = loss_acc_calc(test_dataloader)
print(f"Test Accuracy: {(correct / total) * 100.0}%")
wandb.log({'test_accuracy': (correct / total) * 100.0})
# calculate the metrics
y_true, y_pred = evaluation_metris_calc(test_dataloader)
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
# logging the metrics onto wandb
wandb.run.summary["test_accuracy"] = (correct / total) * 100.0
wandb.run.summary["f1_score"] = f1
wandb.run.summary["precision"] = precision
wandb.run.summary["recall"] = recall

# %% [markdown]
# #### Confusion Matrix 

# %%
def plotting_confusion_matrix(dataloader, file_name):
    # calculate the confusion matrix using sklearn 
    y_true, y_pred = evaluation_metris_calc(dataloader)
    cm = confusion_matrix(y_true, y_pred)
    # plot the confusion matrix with numbers
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=dataset.classes, yticklabels=dataset.classes,
           title="Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    plt.show()
    # save the confusion matrix 
    fig.savefig(file_name)

# %%
# plot the confusion matrix for the test set
plotting_confusion_matrix(test_dataloader, f"A1_Q2_CNN_Experiment_{run_num}_Confusion_Matrix")

# %% [markdown]
# #### Analysing the misclassified images (BONUS)

# %%
# find misclassified images and print 3 misclassified images for each class
def misclassified_images(dataloader):
    misclassified_images = {i: [] for i in range(len(dataset.classes))}
    for i, sample in enumerate(dataloader, 0):
        image = sample['image']
        label = sample['label']
        image = image.to(device)
        label = torch.tensor(label).to(device)
        forwards = model.forward(image)
        logits = torch.nn.functional.softmax(forwards, dim=1)
        pred = torch.argmax(logits, dim=1)
        label = int(label.cpu().numpy())
        pred = int(pred.cpu().numpy())
        if (label != pred):
            if (len(misclassified_images[label]) < 3):
                misclassified_images[label].append([image, label, pred])
    return misclassified_images

# %%
misclassified_images = misclassified_images(test_dataloader)

# %%
# plot the misclassified images
fig, axes = plt.subplots(len(dataset.classes), 3, figsize=(10, 30))
for i in range(len(dataset.classes)):
    for j in range(3):
        image = transforms.ToPILImage()((((misclassified_images[i][j][0]+1)/2) * 255).squeeze().to(torch.uint8))
        axes[i, j].imshow(image)
        axes[i, j].set_title(f"Predicted: {dataset.get_class_name(misclassified_images[i][j][2])},\n Actual: {dataset.get_class_name(misclassified_images[i][j][1])}")
        axes[i, j].axis('off')
plt.show()

# %%
wandb.finish()

# %%


# %% [markdown]
# ## Q2.3 Fine-tuning a pre-trained model

# %% [markdown]
# #### Dataset Setup

# %%
resnet18_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
train_size = 0.7
val_size = 0.1
test_size = 0.2
batch_size = 16
run_num = 2

# %%
dataset = RussianWildLifeDataset(new_path, composed_transform)

# %%
class_wise_indices = [np.where(np.array(dataset.labels) == i)[0] for i in range(len(dataset.classes))]
class_distribution = [len(indices) for indices in class_wise_indices]
train_class_distribution = [int(train_size * dist) for dist in class_distribution]
val_class_distribution = [int(val_size * dist) for dist in class_distribution]
test_class_distribution = [int(test_size * dist) for dist in class_distribution]

# %%
train_indices = []
val_indices = []
test_indices = []
for i in range(len(dataset.classes)):
    np.random.shuffle(class_wise_indices[i])
    train_indices.extend(np.random.choice(class_wise_indices[i], train_class_distribution[i], replace=False))
    val_indices.extend(np.random.choice(np.setdiff1d(class_wise_indices[i], train_indices), val_class_distribution[i], replace=False))
    test_indices.extend(np.random.choice(np.setdiff1d(class_wise_indices[i], np.concatenate((train_indices, val_indices))), test_class_distribution[i], replace=False))

# %%
train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)
test_data = Subset(dataset, test_indices)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

# %%
train_size = len(train_data)
val_size = len(val_data)
test_size = len(test_data)
print(train_size, val_size, test_size)

# %% [markdown]
# #### WandB Setup

# %%
print(os.getcwd())
os.chdir("..")
os.chdir("..")
print(os.getcwd())

# %%
import wandb
wandb.login()

# %%
model_config = dict(
    epochs=15,
    classes=10,
    batch_size=batch_size,
    optimizer="Adam",
    learning_rate=0.001,
    dataset="Russian WildLife Dataset",
    architecture="Resnet18", 
    data_augmentation="False",
    weight_initialisation="Default")

# %%
wandb.init(
    project = "Computer_Vision_A1",
    config = model_config,
)

# %% [markdown]
# #### Data Visualization

# %%
data_visualization(train_data)

# %%
data_visualization(val_data)

# %% [markdown]
# #### Model Training

# %%
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
optimizer = optim.Adam(model.parameters())

# %%
wandb.watch(model)

# %%
def train_model(model, model_config, loss_fn, optimizer, train_dataloader, val_dataloader):
    running_loss = 0.0
    model.train()
    for epoch in range(model_config['epochs']):
        for i, sample in enumerate(train_dataloader, 0):
            inputs = sample['image']
            labels = sample['label']
            inputs = inputs.to(device)
            labels = torch.tensor(labels).to(device)
            # zero the gradients before backprop
            optimizer.zero_grad()
            # forward pass
            logits = model.forward(inputs)
            loss = loss_fn(logits, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            minibatch_loss = loss.item()
            running_loss += minibatch_loss
            # logging the minibatch after every batch, and running loss after every 100 minibatches
            wandb.log({'minibatch_loss': minibatch_loss})
            if (i%100==0):
                wandb.log({'running_loss': running_loss / 100})
                running_loss = 0.0
        # log the training accuracy and loss
        correct, total, loss = loss_acc_calc(train_dataloader)
        wandb.log({'train_loss': loss / len(train_dataloader), 'train_accuracy': (correct / total) * 100.0})
        # log the validation accuracy and loss
        correct, total, loss = loss_acc_calc(val_dataloader)
        wandb.log({'val_loss': loss / len(val_dataloader), 'val_accuracy': (correct / total) * 100.0})
    # save the model after training
    torch.save(model.state_dict(), f"A1_Q2_ResNet18_Experiment_{run_num}.pth")
    return model

# %%
model = train_model(model, model_config, loss_fn, optimizer, train_dataloader, val_dataloader)

# %% [markdown]
# #### Evaluation 

# %%
# testing the model 
correct, total, loss = loss_acc_calc(test_dataloader)
print(f"Test Accuracy: {(correct / total) * 100.0}%")
wandb.log({'test_accuracy': (correct / total) * 100.0})
# calculate the metrics
y_true, y_pred = evaluation_metris_calc(test_dataloader)
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
# logging the metrics onto wandb
wandb.run.summary["test_accuracy"] = (correct / total) * 100.0
wandb.run.summary["f1_score"] = f1
wandb.run.summary["precision"] = precision
wandb.run.summary["recall"] = recall

# %%
# plot the confusion matrix for the test set
plotting_confusion_matrix(test_dataloader, f"A1_Q2_ResNet18_Experiment_{run_num}_Confusion_Matrix")

# %%
wandb.finish()

# %% [markdown]
# #### Feature Extraction and TSNE Plots
# 

# %%
def feature_extraction(model, dataloader, n_comp):
    model.eval()
    with torch.no_grad():
        features = []
        labels = []
        for i, sample in enumerate(dataloader, 0):
            images = sample['image']
            labels.extend(sample['label'])
            images = images.to(device)
            forwards = model.forward(images)
            features.extend(forwards.cpu().numpy())
        features = np.array(features)
        labels = np.array(labels)
        data_tsne = TSNE(n_components=n_comp)
        features = data_tsne.fit_transform(features)
        return features, labels

# %%
training_features, training_labels = feature_extraction(model, train_dataloader, 2)
val_features, val_labels = feature_extraction(model, val_dataloader, 2)

# %%
plt.figure(figsize=(10, 10))
for i in range(len(dataset.classes)):
    indices = np.where(training_labels == i)
    plt.scatter(training_features[indices, 0], training_features[indices, 1], label=dataset.get_class_name(i))
plt.title("Training Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
plt.savefig(f"A1_Q2_ResNet18_Experiment_{run_num}_Training_TSNE_2D.png")

# %%
training_features, training_labels = feature_extraction(model, train_dataloader, 3)
val_features, val_labels = feature_extraction(model, val_dataloader, 3)

# %%
plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
for i in range(len(dataset.classes)):
    indices = np.where(training_labels == i)
    ax.scatter3D(training_features[indices, 0], training_features[indices, 1], training_features[indices, 2], label=dataset.get_class_name(i))
ax.set_title("Training Set")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
plt.legend()
plt.show()
plt.savefig(f"A1_Q2_ResNet18_Experiment_{run_num}_Training_TSNE_3D.png")

# %% [markdown]
# #### Misclassification Analysis

# %%
# plot the misclassified images
fig, axes = plt.subplots(len(dataset.classes), 3, figsize=(10, 30))
for i in range(len(dataset.classes)):
    for j in range(3):
        image = transforms.ToPILImage()((((misclassified_images[i][j][0]+1)/2) * 255).squeeze().to(torch.uint8))
        axes[i, j].imshow(image)
        axes[i, j].set_title(f"Predicted: {dataset.get_class_name(misclassified_images[i][j][2])},\n Actual: {dataset.get_class_name(misclassified_images[i][j][1])}")
        axes[i, j].axis('off')
plt.show()

# %% [markdown]
# ## Q2.4 Data Augmentation

# %% [markdown]
# #### Dataset Preparation

# %%
composed_transform = transforms.Compose([
    transforms.Resize([224, 224]), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# %%
train_size = 0.7
val_size = 0.1
test_size = 0.2
batch_size = 16
run_num = 2

# %%
dataset = RussianWildLifeDataset(new_path, composed_transform)

# %%
class_wise_indices = [np.where(np.array(dataset.labels) == i)[0] for i in range(len(dataset.classes))]
class_distribution = [len(indices) for indices in class_wise_indices]
train_class_distribution = [int(train_size * dist) for dist in class_distribution]
val_class_distribution = [int(val_size * dist) for dist in class_distribution]
test_class_distribution = [int(test_size * dist) for dist in class_distribution]

# %%
train_indices = []
val_indices = []
test_indices = []
for i in range(len(dataset.classes)):
    np.random.shuffle(class_wise_indices[i])
    train_indices.extend(np.random.choice(class_wise_indices[i], train_class_distribution[i], replace=False))
    val_indices.extend(np.random.choice(np.setdiff1d(class_wise_indices[i], train_indices), val_class_distribution[i], replace=False))
    test_indices.extend(np.random.choice(np.setdiff1d(class_wise_indices[i], np.concatenate((train_indices, val_indices))), test_class_distribution[i], replace=False))

# %%
train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)
test_data = Subset(dataset, test_indices)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

# %% [markdown]
# #### Data Augmentation

# %%
augmentation_transform1 = transforms.Compose([
    transforms.Resize([256, 256]), 
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(30),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# %%
# apply augmentation transforms to current train data and create new train dataset
augmentation_train_data = []
for sample in train_data:
    image = sample['image']
    label = sample['label']
    augmentation_train_data.append({'image': augmentation_transform1(image), 'label': label})
augmentation_train_data = torch.utils.data.ConcatDataset([train_data, augmentation_train_data])
augmentation_train_dataloader = DataLoader(augmentation_train_data, batch_size=batch_size, shuffle=True)

# %% [markdown]
# #### Data Visualization

# %%
plot_distribution(augmentation_train_data, 'Augmented Train set')
plot_distribution(train_data, 'Original Train set')
plot_distribution(val_data, 'Validation set')

# %%
augmented_train_size = len(augmentation_train_data)
original_train_size = len(train_data)
val_size = len(val_data)
test_size = len(test_data)
print(augmented_train_size, original_train_size, val_size, test_size)

# %% [markdown]
# #### WandB Setup

# %%
print(os.getcwd())
os.chdir("..")
os.chdir("..")
print(os.getcwd())

# %%
import wandb
wandb.login()

# %%
model_config = dict(
    epochs=15,
    classes=10,
    batch_size=batch_size,
    optimizer="Adam",
    learning_rate=0.001,
    dataset="Russian WildLife Dataset",
    architecture="ResNet18",
    data_augmentation="True",
    weight_initialisation="Default")

# %%
wandb.init(
    project = "Computer_Vision_A1",
    config = model_config,
)

# %% [markdown]
# #### Dataset Visualization

# %%
data_visualization(augmentation_train_data)

# %%
data_visualization(train_data)

# %%
data_visualization(val_data)

# %%
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
optimizer = optim.Adam(model.parameters())

# %%
wandb.watch(model)

# %% [markdown]
# #### Model Training

# %%
def train_model_augmented(model, model_config, loss_fn, optimizer, augmented_dataloader, train_dataloader, val_dataloader):
    running_loss = 0.0
    model.train()
    for epoch in range(model_config['epochs']):
        for i, sample in enumerate(augmented_dataloader, 0):
            inputs = sample['image']
            labels = sample['label']
            inputs = inputs.to(device)
            labels = torch.tensor(labels).to(device)
            # zero the gradients before backprop
            optimizer.zero_grad()
            # forward pass
            logits = model.forward(inputs)
            loss = loss_fn(logits, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            minibatch_loss = loss.item()
            running_loss += minibatch_loss
            # logging the minibatch after every batch, and running loss after every 100 minibatches
            wandb.log({'minibatch_loss': minibatch_loss})
            if (i%100==0):
                wandb.log({'running_loss': running_loss / 100})
                running_loss = 0.0
        # log the training accuracy and loss
        correct, total, loss = loss_acc_calc(train_dataloader)
        wandb.log({'train_loss': loss / len(train_dataloader), 'train_accuracy': (correct / total) * 100.0})
        # log the validation accuracy and loss
        correct, total, loss = loss_acc_calc(val_dataloader)
        wandb.log({'val_loss': loss / len(val_dataloader), 'val_accuracy': (correct / total) * 100.0})
    # save the model after training
    torch.save(model.state_dict(), f"A1_Q2_ResNet_DAG_Experiment_1.pth")
    return model

# %%
model = train_model_augmented(model, model_config, loss_fn, optimizer, augmentation_train_dataloader, train_dataloader, val_dataloader)

# %% [markdown]
# #### Evaluation

# %%
# testing the model 
correct, total, loss = loss_acc_calc(test_dataloader)
print(f"Test Accuracy: {(correct / total) * 100.0}%")
wandb.log({'test_accuracy': (correct / total) * 100.0})
# calculate the metrics
y_true, y_pred = evaluation_metris_calc(test_dataloader)
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
# logging the metrics onto wandb
wandb.run.summary["test_accuracy"] = (correct / total) * 100.0
wandb.run.summary["f1_score"] = f1
wandb.run.summary["precision"] = precision
wandb.run.summary["recall"] = recall

# %%
# plot the confusion matrix for the test set
plotting_confusion_matrix(test_dataloader, f"A1_Q2_ResNet_DAG_Experiment_{run_num}_Confusion_Matrix")

# %%
wandb.finish()

# %% [markdown]
# ## Q2.5 BONUS

# %%
CNN_model = ConvNet()
CNN_model.load_state_dict(torch.load('A1_Q2_CNN_Experiment_2.pth'))
CNN_model.to(device)
ResNet_Model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
ResNet_Model.fc = nn.Linear(ResNet_Model.fc.in_features, 10)
ResNet_Model.load_state_dict(torch.load('A1_Q2_ResNet18_Experiment_1.pth'))
ResNet_Model.to(device)
ResNet_DAG_Model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
ResNet_DAG_Model.fc = nn.Linear(ResNet_DAG_Model.fc.in_features, 10)
ResNet_DAG_Model.load_state_dict(torch.load('A1_Q2_ResNet_DAG_Experiment_1.pth'))
ResNet_DAG_Model.to(device)

# %%
class_wise_misclassified = {i: [] for i in range(len(dataset.classes))}
for i, sample in enumerate(test_dataloader, 0):
    image = sample['image']
    label = sample['label']
    image = image.to(device)
    label = torch.tensor(label).to(device)
    forwards = CNN_model.forward(image)
    logits = torch.nn.functional.softmax(forwards, dim=1)
    pred = torch.argmax(logits, dim=1)
    label = int(label.cpu().numpy())
    pred = int(pred.cpu().numpy())
    if (label != pred):
        if (len(class_wise_misclassified[label]) < 3):
            class_wise_misclassified[label].append([image, forwards, label, pred])

# %%
# finding the centroid of each class
class_centroids = []
with torch.no_grad():
    for i in range(len(dataset.classes)):
        class_images = []
        for sample in test_data:
            if (sample['label'] == i):
                class_images.append(sample['image'])
        class_images = torch.stack(class_images)
        forwards = CNN_model.forward(class_images.to(device))
        class_centroids.append(torch.mean(forwards, dim=0))
class_centroids = torch.stack(class_centroids)
print(class_centroids)

# %%
for i in class_wise_misclassified.keys():
    for j in range(3):
        image = class_wise_misclassified[i][j][0]
        forwards = class_wise_misclassified[i][j][1]
        label = class_wise_misclassified[i][j][2]
        pred = class_wise_misclassified[i][j][3]
        # ResNet18 
        forwards_resnet = ResNet_Model.forward(image)
        logits_resnet = torch.nn.functional.softmax(forwards_resnet, dim=1)
        pred_resnet = torch.argmax(logits_resnet, dim=1)
        # ResNet18 DAG
        forwards_resnet_dag = ResNet_DAG_Model.forward(image)
        logits_resnet_dag = torch.nn.functional.softmax(forwards_resnet_dag, dim=1)
        pred_resnet_dag = torch.argmax(logits_resnet_dag, dim=1)
        # distance from the centroid
        cnn_distance_gt = torch.dist(forwards, class_centroids[label])
        cnn_distance_pred = torch.dist(forwards, class_centroids[pred])
        resnet_distance_gt = torch.dist(forwards_resnet, class_centroids[label])
        resnet_distance_pred = torch.dist(forwards_resnet, class_centroids[pred])
        resnet_dag_distance_gt = torch.dist(forwards_resnet_dag, class_centroids[label])
        resnet_dag_distance_pred = torch.dist(forwards_resnet_dag, class_centroids[pred])
        label = int(label)
        # CNN
        print("CNN")
        print("Predicted: ", dataset.get_class_name(pred), ", Actual: ", dataset.get_class_name(label))
        print("Distance from ground truth: ", np.round(cnn_distance_gt.item(), 2))
        print("Distance from prediction: ", np.round(cnn_distance_pred.item(), 2))
        # ResNet18
        if (pred_resnet == label):
            resnet_distance_gt = 0
        print("ResNet18")
        print("Predicted: ", dataset.get_class_name(int(pred_resnet.cpu())), ", Actual: ", dataset.get_class_name(label))
        print("Distance from ground truth: ", np.round(resnet_distance_gt.item(), 2))
        print("Distance from prediction: ", np.round(resnet_distance_pred.item(), 2))
        # ResNet18 DAG
        if (pred_resnet_dag == label):
            resnet_dag_distance_gt = 0
        print("ResNet18 DAG")
        print("Predicted: ", dataset.get_class_name(int(pred_resnet_dag.cpu())), ", Actual: ", dataset.get_class_name(label))
        print("Distance from ground truth: ", np.round(resnet_dag_distance_gt.item(), 2))
        print("Distance from prediction: ", np.round(resnet_dag_distance_pred.item(), 2))
        image = class_wise_misclassified[i][j][0]
        image = transforms.ToPILImage()((((image+1)/2) * 255).squeeze().to(torch.uint8))
        # ResNet18 DAG
        print()
        plt.imshow(image)
        plt.show()
        


