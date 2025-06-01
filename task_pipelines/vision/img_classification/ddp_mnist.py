import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torchmetrics
from torchmetrics.classification import MulticlassAccuracy

import sys
import os
import re
import wandb
import random
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

def set_torch_backends_ampere():
    """
    Ampare architecture : 30xx, a100, h100,..
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



def fix_random_seed(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    # 실행시 torchrun or torch.distributed.launch --
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"]) # dist.get_rank()
        args.world_size = int(os.environ['WORLD_SIZE']) # dist.get_world_size()
        args.local_rank = int(os.environ['LOCAL_RANK']) # args.rank % torch.cuda.device_count()
    # 스크립트 실행시
    # #SBATCH --gres=gpu:x
    # #SBATCH --ntasks-per-node=x
    # python train.py
    # python train.py를 x번 돌리는 경우
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.node_rank = int(os.environ['SLURM_NODEID'])

        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['RANK'] = str(args.rank)
    # 스크립트 실행시 : 슬럼 옵션 사용하지 않을시
    # 이때는 torchrun or torch.distributed.launch도 안쓴다고 가정
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.local_rank, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        is_master = True
    # GPU 마저도 안쓰는 경우 그냥 종료
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_filename", type=str, default="checkpoint.pth")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mnist_cnn_pytorch")
    parser.add_argument("--wandb_entity", type=str, default="ty-kim")
    parser.add_argument("--wandb_key", type=str)
    args = parser.parse_args()

    # ============================ Distributed Setting ============================
    init_distributed_mode(args)

    # ============================ Basic Setting ==================================
    if args.wandb and args.rank == 0:
        wandb.login(key=args.wandb_key, force=True)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    # 전 rank가 완전히 동일한 시드(예: seed=42)를 써도, DistributedSampler에서 랜덤하게 데이터를 분배하기 때문에 문제 없음
    # 모두 동일한 랜덤 시퀀스로 augmentation이 적용될 가능성이 생기는 등 원치않는 효과 방지
    fix_random_seed(args.seed+args.rank)
    # ============================ Dataset =========================================
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1)

    if args.precision == "fp16":
        ptdtype = torch.float16
    elif args.precision == "bf16":
        ptdtype = torch.bfloat16
    else:
        ptdtype = torch.float32

    if args.device.type == 'cuda':
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    else:
        autocast_ctx = torch.autocast(device_type='cpu', dtype=ptdtype)
    
    # Underfitting을 방지
    scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == 'float16'))

    # ============================ Resume setup ========================================
    start_epoch = 0
    if args.resume:
        ckpt_filepath = os.path.join(args.ckpt_dir, args.ckpt_filename)
        # 체크포인트에서 텐서와 메타데이터 가져옴
        # 텐서안에 storage라는 오브젝트가 있는데 이것의 위치를 장치로 옮겨줌
        map_location = lambda storage, loc: storage.cuda(args.local_rank)
        checkpoint = torch.load(ckpt_filepath, map_location=map_location) # = torch.load(ckpt_filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        print(f"Resuming from epoch {start_epoch}", flush=True)


    # ============================ Train ==============================================
    train_acc_metric = MulticlassAccuracy(num_classes=10).to(args.local_rank)
    val_acc_metric = MulticlassAccuracy(num_classes=10).to(args.local_rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_acc_metric.reset()


        if args.rank == 0:
            train_loader = tqdm(train_loader, desc = f'Epoch {epoch}', leave=False)
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to('cuda', pin_memory=True, non_blocking=True)
            target = target.to('cuda', pin_memory=True, non_blocking=True)
            optimizer.zero_grad(set_to_none=True) # for param in model.parameters(): param.grad = None

            with autocast_ctx:
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward() # if scaler is None, this is equivalent to loss.backward()
            scaler.step(optimizer) # if scaler is None, this is equivalent to optimizer.step()
            scaler.update() # if scaler is None, this is equivalent to None


            train_acc_metric.update(output, target)

            if (batch_idx % 100 == 0) and (args.rank == 0) and args.wandb:
                wandb.log({"loss": loss.item()})
                images = wandb.Image(data[0].detach().cpu(), caption=f"Label: {target[0].item()}")
                wandb.log({'examples' : images})

            if args.local_rank == 0:
                train_loader.set_description(f'loss: {loss.item():.4f}')


        scheduler.step()
        # 에포크 끝나고, 전체 training accuracy 계산
        train_acc = train_acc_metric.compute()

        # test
        model.eval()
        val_acc_metric.reset()
        if args.rank == 0:
            pbar = tqdm(test_loader, colour='green', desc=f'Validation Epoch {epoch}')
        else:
            pbar = test_loader

        with torch.no_grad():
            for data, target in pbar:
                data = data.to('cuda', pin_memory=True, non_blocking=True)
                target = target.to('cuda', pin_memory=True, non_blocking=True)

                
                with autocast_ctx:
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
                'scheduler_state_dict' : scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(args.ckpt_dir, f'checkpoint_{epoch}.pth'))


    
    # ============================ Finish ==============================================
    dist.barrier()
    init_end_event.record()
    if args.wandb and args.rank == 0:
        wandb.finish()
    if args.rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
    dist.destroy_process_group()