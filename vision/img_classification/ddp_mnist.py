import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
import os
import re
import wandb
import argparse
import numpy as np
from tqdm import tqdm

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(50176, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def hw_setup_ampare():
    """
    Ampare architecture : 30xx, a100, h100,..
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def resolve_root_node_address(nodes: str) -> str:
    """The node selection format in SLURM supports several formats.

    This function selects the first host name from

    - a space-separated list of host names, e.g., 'host0 host1 host3' yields 'host0' as the root
    - a comma-separated list of host names, e.g., 'host0,host1,host3' yields 'host0' as the root
    - the range notation with brackets, e.g., 'host[5-9]' yields 'host5' as the root

    """
    nodes = re.sub(r"\[(.*?)[,-].*\]", "\\1", nodes)  # Take the first node of every node range
    nodes = re.sub(r"\[(.*?)\]", "\\1", nodes)  # handle special case where node range is single number
    return nodes.split(" ")[0].split(",")[0]

def get_main_address():
    root_node = os.environ.get("MASTER_ADDR")
    if root_node is None:
        nodelist = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        root_node = resolve_root_node_address(nodelist)
        os.environ["MASTER_ADDR"] = root_node

    return root_node

def init_distributed_mode(args):
    """
    from DoRA
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"]) # dist.get_rank()
        args.world_size = int(os.environ['WORLD_SIZE']) # dist.get_world_size()
        args.local_rank = int(os.environ['LOCAL_RANK']) # args.rank % torch.cuda.device_count()
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.node_rank = int(os.environ['SLURM_NODEID'])

        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['RANK'] = str(args.rank)

    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.local_rank, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        is_master = True
    else:
        print('Not using distributed mode')
        sys.exit(1)

    os.environ['MASTER_ADDR'] = get_main_address()
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend='nccl', 
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.local_rank)
    is_master = args.rank == 0
    dist.barrier()

    setup_for_distributed(args.rank == 0)
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mnist_cnn_pytorch")
    parser.add_argument("--wandb_entity", type=str, default="ty-kim")
    parser.add_argument("--wandb_key", type=str)
    args = parser.parse_args()

    init_distributed_mode(args)

    wandb.login(key=args.wandb_key, force=True)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))]
    )

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, shuffle=True)


    model = SimpleCNN().cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,  # this ensures that tensors are sent to the GPU
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(args.epochs):
        model.train()
        if args.local_rank == 0:
            train_loader = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            if batch_idx % 10 == 0:
                wandb.log({"loss": loss.item()})
                images = wandb.Image(data[0], caption=f"Label {target[0]}")
                wandb.log({'examples' : images})

            if args.local_rank == 0:
                train_loader.set_description(f'loss: {loss.item():.4f}')

    wandb.finish()