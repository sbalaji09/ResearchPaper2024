import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import cv2
from tensorflow.keras import layers, models
import torch.optim as optim

from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, Dataset, random_split
import tensorflow as tf
from PIL import Image

import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

NUM_CLIENTS = 5
BATCH_SIZE = 32

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {fl.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

class AlexNet(nn.Module):
    def __init__(self, input_shape, num_classes, **kwargs):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        self._load_images_and_labels()
    
    def _load_images_and_labels(self):
        unique_labels = sorted(set(label_name for label_name in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, label_name))))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        for label_name in unique_labels:
            label_dir = os.path.join(self.directory, label_name)
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(image_path)
                    self.labels.append(self.label_map[label_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        
        label = torch.tensor(label, dtype=torch.long) 
        return {"img": image, "label": label}

class FederatedDataset:
    def __init__(self, dataset_dir, partitioners):
        self.dataset_dir = dataset_dir
        self.partitioners = partitioners

    def load_partition(self, partition_id, split):
        partition_dir = os.path.join(self.dataset_dir, split)
        partition = CustomDataset(partition_dir)
        return partition

    def load_split(self, split):
        split_dir = os.path.join(self.dataset_dir, split)
        return CustomDataset(split_dir)

def prepare_data():
    ben_train_dir = '/Users/siddharthbalaji/Documents/Lumiere_Coding/archive/train/benign/'
    mal_train_dir = '/Users/siddharthbalaji/Documents/Lumiere_Coding/archive/train/malignant/'


    ben_test_dir = '/Users/siddharthbalaji/Documents/Lumiere_Coding/archive/test/benign/'
    mal_test_dir = '/Users/siddharthbalaji/Documents/Lumiere_Coding/archive/test/malignant/'


    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for filename in os.listdir(ben_train_dir):
        img = cv2.imread(ben_train_dir + filename)
        img = cv2.resize(img, (50, 50))
        train_data.append(img)
        train_labels.append(0)  

    for filename in os.listdir(mal_train_dir):
        img = cv2.imread(mal_train_dir + filename)
        img = cv2.resize(img, (50, 50))
        train_data.append(img)
        train_labels.append(1) 

    for filename in os.listdir(ben_test_dir):
        img = cv2.imread(ben_test_dir + filename)
        img = cv2.resize(img, (50, 50))
        test_data.append(img)
        test_labels.append(0)  

    for filename in os.listdir(mal_test_dir):
        img = cv2.imread(mal_test_dir + filename)
        img = cv2.resize(img, (50, 50))
        test_data.append(img)
        test_labels.append(1)  

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))


    print(len(train_ds))
    print(len(test_ds))

    def process_image(image,label):
        image=tf.image.per_image_standardization(image)
        image=tf.image.resize(image,(64,64))
        
        return image,label

    train_ds_size=tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size=tf.data.experimental.cardinality(test_ds).numpy()
    print('Train size:',train_ds_size)
    print('Test size:',test_ds_size)

    train_ds=(train_ds
            .map(process_image)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=32,drop_remainder=True)
            )
    test_ds=(test_ds
            .map(process_image)
            .shuffle(buffer_size=test_ds_size)
            .batch(batch_size=32,drop_remainder=True)
            )
    return train_ds, test_ds

def train(model, trainloader, epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Use the appropriate optimizer

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            inputs, labels = batch['img'].to(DEVICE), batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader)}")

    return model

def test(model, testloader):

    
    model.eval() 
    criterion = nn.CrossEntropyLoss() 

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): 
        for batch in testloader:
            inputs, labels = batch['img'].to(DEVICE), batch['label'].to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(testloader)
    accuracy = correct / total

    return avg_loss, accuracy


# model = AlexNet((64,64,3), 2)
# train_ds, test_ds = prepare_data()
# history = train(model, train_ds, test_ds, epochs=5)
# print('Accuracy Score = ', np.max(history.history['val_accuracy']))

# test_results = test(model, test_ds)


###############################################################################################################################################################

def load_datasets():
    dataset_dir = '/Users/siddharthbalaji/Documents/Lumiere_Coding/archive'
    fds = FederatedDataset(dataset_dir, partitioners={"train": NUM_CLIENTS})

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def apply_transforms(dataset):
        dataset.transform = transform
        return dataset

    def split_dataset(dataset, train_size=0.8):
        total_size = len(dataset)
        train_size = int(total_size * train_size)
        val_size = total_size - train_size
        return random_split(dataset, [train_size, val_size])

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        partition = apply_transforms(partition)
        train_set, val_set = split_dataset(partition, train_size=0.8)
        trainloaders.append(DataLoader(train_set, batch_size=BATCH_SIZE))
        valloaders.append(DataLoader(val_set, batch_size=BATCH_SIZE))
    
    testset = fds.load_split("test")
    testset = apply_transforms(testset)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    
    return trainloaders, valloaders, testloader


trainloaders, valloaders, testloader = load_datasets()

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config.get("server_round", 0)
        local_epochs = config.get("local_epochs", 1)
        
        # Use values provided by the config
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=local_epochs)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    

def client_fn(cid) -> FlowerClient:
    net = AlexNet((64,64,3), 2).to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = AlexNet((64,64,3), 2).to(DEVICE)
    valloader = valloaders[0]
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

def fit_config(server_round: int) -> Dict[str, int]:
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    return {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 2 if server_round < 2 else 5,
    }


strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(AlexNet((64,64,3), 2))),
    evaluate_fn=evaluate,   
    on_fit_config_fn=fit_config,  # Pass the fit_config function
    evaluate_metrics_aggregation_fn=weighted_average,  # Pass the weighted_average function
)


client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}

# Start the simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
    client_resources=client_resources,
)