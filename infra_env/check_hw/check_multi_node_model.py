"""
Multi-Node Rank 분배되는지 체크
"""
import argparse
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_epochs')
    # parser.add_argument('--batch_size')
    # parser.add_argument('--learning_rate')
    # parser.add_argument('--model_dir')
    # parser.add_argument("--model_filename")
    # parser.add_argument("--resume", action="store_true")
    parser.add_argument('--sleep', type=int, default=10)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)

    print('global_rank:', global_rank, 'local_rank :', local_rank, 'world_size :', world_size, 'device:', torch.device('cuda'))

    model = Net().to('cuda')
    model = DDP(model, device_ids=[self.local_rank])
    
    time.sleep(args.sleep)
    