import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier

import torchvision
from torchvision import datasets, transforms
import yaml
from pathlib import Path

class Hyperparams:
    random_seed = 123
    batch_size = 32
    test_batch_size = 32
    lr = 1e-3
    epochs = 10
    save_model = False
    log_interval = 100
    model_dir = './model'
    model_filename = 'model.pth'
    
class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout(0.25)
      self.dropout2 = nn.Dropout(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = F.max_pool2d(x, 2)
      x = self.dropout1(x)
      x = torch.flatten(x, 1)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)
      output = F.log_softmax(x, dim=1)
      return output


def train(args, model, train_loader, optimizer, epoch, local_rank, global_rank):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device=local_rank), target.to(device=local_rank)#data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('[GPU{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(global_rank, epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item())
                )
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single-parallel')
    parser.add_argument('yaml', type=str)
    args = parser.parse_args()

    init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)

    torch.manual_seed(Hyperparams.random_seed)

    try:
        yaml_dict = yaml.safe_load(Path(args.yaml).read_text())
        for key, value in yaml_dict.items():
            setattr(Hyperparams, key, value)
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    dataset1 = datasets.MNIST('/purestorage/slurm_test/data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset1)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=Hyperparams.batch_size, shuffle=False, sampler=sampler)

    model = Net().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=float(yaml_dict['lr']))
    for epoch in range(0, 10):
        train_loader.sampler.set_epoch(epoch)
        train(Hyperparams, model, train_loader, optimizer, epoch, local_rank, global_rank)
        
    if Hyperparams.save_model and global_rank == 0:
        os.makedirs(Hyperparams.model_dir, exist_ok=True)
        model_path = os.path.join(Hyperparams.model_dir, Hyperparams.model_filename)
        torch.save(model.state_dict(), model_path)
        
    destroy_process_group()