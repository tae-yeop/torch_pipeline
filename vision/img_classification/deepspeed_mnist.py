import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import deepspeed
from torch.utils.data import DataLoader

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == "__main__":
    
    torch.cuda.set_device(local_rank)



    deepspeed.init_distributed(distributed_backend="nccl", init_method="env://")

    model = MNISTModel()
    model_engine, optimizer, trainloader, sch

    num_epoch = 10
    for epoch in range(epoch):
        model_engine.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
