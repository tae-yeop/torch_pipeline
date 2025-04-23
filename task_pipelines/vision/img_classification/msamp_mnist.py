import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset


from torchvision import datasets, transforms

import torchmetrics
from torchmetrics.classification import MulticlassAccuracy

import sys
import os
import re
import argparse
import numpy as np
from tqdm import tqdm

import msamp
from msamp import deepspeed


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
    args.device = torch.device('cuda', args.local_rank)

    is_master = args.rank == 0
    dist.barrier()

    setup_for_distributed(args.rank == 0)
    print('| distributed init (rank {}): {}'.format(args.rank, os.environ['MASTER_ADDR']), flush=True)



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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    # ============================ Distributed Setting ============================
    init_distributed_mode(args)

    # ============================ Basic Setting ==================================
    fix_random_seeds(args.seed+args.rank)
    assert args.batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))]
    )

    if dist.get_rank() != 0:
        dist.barrier()

    train_dataset_tmp = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_tmp, [50000, 10000])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if dist.get_rank() == 0:
        dist.barrier()


    if args.world_size > 1:
        # DistributedSampler의 디폴트 shuffle는 True임
        train_sampler = DistributedSampler(
            train_dataset, 
            rank=args.rank, 
            num_replicas=args.world_size, 
            shuffle=True
        )
        test_sampler = DistributedSampler(
            test_dataset, 
            rank=args.rank, 
            num_replicas=args.world_size, 
            shuffle=False
            )
    else:
        train_sampler, test_sampler = None, None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                              num_workers=4, pin_memory=True, shuffle=(train_sampler is None))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, 
                             num_workers=4, pin_memory=True, shuffle=False)

    # ============================ Models =========================================
    model = SimpleCNN().cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,  # this ensures that tensors are sent to the GPU
    )
    
    # ============================ Traning setup ======================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model, optimizer = msamp.initialize(model, optimizer, opt_level="O2")

    # ============================ Train ==============================================
    train_acc_metric = MulticlassAccuracy(num_classes=10).to(args.local_rank)
    val_acc_metric = MulticlassAccuracy(num_classes=10).to(args.local_rank)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_acc_metric.reset()
        
        if args.rank == 0:
            train_loader = tqdm(train_loader, desc = f'Epoch {epoch}', leave=False)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to('cuda', pin_memory=True, non_blocking=True)
            target = target.to('cuda', pin_memory=True, non_blocking=True)

            optimizer.zero_grad()

            # -------------------------------------------------
            # msamp.autocast() 영역에서 FP16/BF16/FP8 연산 수행
            # -------------------------------------------------
            with msamp.autocast():
                output = model(data)
                loss = criterion(output, target)

            msamp.scaler.scale(loss).backward()

            # step
            msamp.scaler.step(optimizer)
            msamp.scaler.update()

            train_acc_metric.update(output, target)

            if args.local_rank == 0:
                train_loader.set_description(f'loss: {loss.item():.4f}')

        train_acc = train_acc_metric.compute()

        # test
        model.eval()
        val_acc_metric.reset()
        pbar = tqdm(range(len(test_loader)), colour='green', desc='Validation Epoch')

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to('cuda', pin_memory=True, non_blocking=True)
                target = target.to('cuda', pin_memory=True, non_blocking=True)
                
                with msamp.autocast():
                    output = model(data)
                    val_loss = criterion(output, target)
                
                val_acc_metric.update(output, target)


                if args.rank == 0:
                    pbar.update(1)

        if args.rank == 0:
            pbar.close()

        val_acc = val_acc_metric.compute()

        # save checkpoint
        if args.rank == 0:
            print(f"[Epoch {epoch}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(), # 이렇게 해야 나중에 싱글 추론 가능
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'checkpoint_{epoch}.pth')
    # ============================ Finish ==============================================
    dist.barrier()
    dist.destroy_process_group()
