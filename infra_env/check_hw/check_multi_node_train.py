"""
참조 
https://github.com/WongKinYiu/yolov7/blob/main/train.py
https://github.com/CUBOX-Co-Ltd/cifar10_DDP_FSDP_DS
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as tvm
import torchvision.transforms as transforms


# Ampare architecture 30xx, a100, h100,..
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# infernce
# torch.set_grad_enabled(False)


def train(model, train_loader, optimizer, criterion, epoch, rank, scaler=None, sampelr=None):
    model.train()
    ddp_loss = torch.zeros(2).to('cuda')
    if sampler:
        sampler.set_epoch(epoch)
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to('cuda', pin_memory=True, non_blocking=True)
        labels = labels.to('cuda', pin_memory=True, non_blocking=True)
        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)
    if scaler:
        loss = _grad_scaler.scale(loss)
        loss.backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

# 디폴트 init_method 'env://' 사용
dist.init_process_group(backend='nccl')
global_rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)


# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    trnasforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


if dist.get_rank() != 0:
    dist.barrier()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data',train=False, download=True, transform=transform)

if dist.get_rank() == 0:
    dist.barrier()

batch_size = 512
train_sampler = DistributedSampler(trainset)
test_sampler = DistributedSampler(testset, shuffle=False)
train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
test_loader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
model.to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


for epoch in range(10):
    train(model, train_loader, optimizer, criterion, epoch, global_rank, train_sampler)
    scheduler.step()

dist.barrier()