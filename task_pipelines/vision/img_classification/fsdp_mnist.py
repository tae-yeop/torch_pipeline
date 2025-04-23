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
import argparse
import numpy as np
from tqdm import tqdm

# ============ [FSDP IMPORTS] ============
# ---- [FSDP 관련: PyTorch 2.x 권장 방식] ----
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
    ShardingStrategy,
    summon_full_params,              # <-- PyTorch 2.x 권장
    optim_state_dict,                # <-- optimizer state를 full 형태로 뽑아내는 함수
    load_optim_state_dict
)

# summon_full_params
# FSDP.state_dict_type + FullStateDictConfig를 사용하여 full/sharded state_dict를 구분했는데,
# 2.x 버전부터는 좀 더 직관적인 FSDP.summon_full_params(model) 컨텍스트 매니저가 추가되었습니다.
# 이 컨텍스트 내에서 model.state_dict() / model.load_state_dict()를 호출하면 
# 각 rank에 분산되어 있던 파라미터를 모아(full) 접근할 수 있으므로, 체크포인트 저장/로딩이 쉬워

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


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
    
def set_torch_backends_ampare():
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
    """Select the first host name from node list notation."""
    nodes = re.sub(r"\[(.*?)[,-].*\]", "\\1", nodes)  
    nodes = re.sub(r"\[(.*?)\]", "\\1", nodes)
    return nodes.split(" ")[0].split(",")[0]

def get_main_address():
    root_node = os.environ.get("MASTER_ADDR")
    if root_node is None:
        nodelist = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        root_node = resolve_root_node_address(nodelist)
        os.environ["MASTER_ADDR"] = root_node
    return root_node

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"]) 
        args.world_size = int(os.environ['WORLD_SIZE']) 
        args.local_rank = int(os.environ['LOCAL_RANK']) 
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

    setup_for_distributed(args.rank == 0)
    dist.barrier()
    print('| distributed init (rank {}): {}'.format(args.rank, os.environ['MASTER_ADDR']), flush=True)

# ============ [FSDP-RELATED: Save/Load funcs] ============
def save_checkpoint_fsdp(model, optimizer, epoch, scheduler, ckpt_path, rank=0):
    """
    [FSDP] summon_full_params로 전체 파라미터를 모은 뒤 state_dict 저장
    PyTorch 2.x FSDP 최신 API를 사용해 체크포인트 저장하는 예시
    """
    if rank != 0:
        return  # 실질적으로 rank=0에서만 저장

    # 1) full model state dict
    with summon_full_params(model):
        full_model_sd = model.state_dict()

    # 2) full optimizer state dict
    full_optim_sd = optim_state_dict(model, optimizer)

    ckpt = {
        'epoch': epoch,
        'model_state_dict': full_model_sd,
        'optimizer_state_dict': full_optim_sd,
        'scheduler_state_dict': scheduler.state_dict()
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"[rank={rank}] Saved checkpoint to {ckpt_path}")


def load_checkpoint_fsdp(model, optimizer, scheduler, ckpt_path, rank=0):
    """
    [FSDP] summon_full_params로 full state dict를 load
    PyTorch 2.x FSDP 최신 API를 사용해 체크포인트 로드하는 예시
    """
    dist.barrier()
    if rank == 0:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    else:
        ckpt = None
    dist.barrier()

    # rank 0이 로드한 체크포인트를 다른 rank와 공유는 하지 않고,
    # FSDP API를 이용해 full state dict를 summon_full_params로 로드
    with summon_full_params(model):
        if rank == 0:
            model.load_state_dict(ckpt['model_state_dict'])

    # summon_full_params ctx manager가 끝나면 각 rank의 model state dict는 sharded 형태로 돌아옴
    dist.barrier()

    # Optimizer state 로드
    if rank == 0:
        full_optim_sd = ckpt['optimizer_state_dict']
    else:
        full_optim_sd = None
    load_optim_state_dict(model, optimizer, full_optim_sd)
    dist.barrier()

    if rank == 0:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"[rank=0] Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    return start_epoch


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

    # ============================ Distributed Init ============================
    init_distributed_mode(args)

    # ============================ Basic Setting ==============================
    if args.wandb and args.rank == 0:
        wandb.login(key=args.wandb_key, force=True)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    # Seed
    fix_random_seeds(args.seed + args.rank)

    # ============================ Dataset ===================================
    assert args.batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if dist.get_rank() != 0:
        dist.barrier()
    train_dataset_tmp = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_tmp, [50000, 10000])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    if dist.get_rank() == 0:
        dist.barrier()

    
    if args.world_size > 1:
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

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=4,
                              pin_memory=True,
                              shuffle=(train_sampler is None))
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             sampler=test_sampler,
                             num_workers=4,
                             pin_memory=True,
                             shuffle=False)
    
    # ============================ Model & FSDP ==============================
    model = SimpleCNN().cuda()

    # ============ [FSDP Setup] ============
    # FSDP MixedPrecision 설정
    if args.precision == "fp16":
        param_dtype = torch.float16
        reduce_dtype = torch.float16
        buffer_dtype = torch.float16
    elif args.precision == "bf16":
        param_dtype = torch.bfloat16
        reduce_dtype = torch.bfloat16
        buffer_dtype = torch.bfloat16
    else:  # fp32
        param_dtype = torch.float32
        reduce_dtype = torch.float32
        buffer_dtype = torch.float32

    mp_policy = MixedPrecision(param_dtype=param_dtype,
                               reduce_dtype=reduce_dtype,
                               buffer_dtype=buffer_dtype)
    
    # [FSDP] 모델 래핑
    fsdp_config = {
        "mixed_precision": mp_policy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "cpu_offload": CPUOffload(offload_params=False),
        "device_id": args.local_rank,
    }

    model = FSDP(model, **fsdp_config)

    # ====================== Training Setup =========================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1)

    # ---- GradScaler: FSDP에서는 ShardedGradScaler 권장 ----
    # enabled 여부: fp16이면 True, 나머지는 False
    scaler = ShardedGradScaler(enabled=(args.precision == "fp16"))

    # ---- autocast_ctx 유지 (원본 코드 스타일) ----
    if args.device.type == 'cuda':
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=param_dtype)
    else:
        # CPU FSDP 시
        autocast_ctx = torch.amp.autocast(device_type='cpu', dtype=param_dtype)

    # ============================ Resume ====================================
    start_epoch = 0
    if args.resume:
        ckpt_filepath = os.path.join(args.ckpt_dir, args.ckpt_filename)
        if os.path.isfile(ckpt_filepath):
            start_epoch = load_checkpoint_fsdp(
                model, optimizer, scheduler,
                ckpt_path=ckpt_filepath,
                rank=args.rank
            )
        else:
            print(f"[rank={args.rank}] No checkpoint found at {ckpt_filepath}")

    # ============================ Train Loop ================================
    train_acc_metric = MulticlassAccuracy(num_classes=10).to(args.device)
    val_acc_metric = MulticlassAccuracy(num_classes=10).to(args.device)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_acc_metric.reset()

        if args.rank == 0:
            train_loader = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # ---- autocast ----
            with autocast_ctx:
                output = model(data)
                loss = criterion(output, target)

            # ---- GradScaler ----
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_acc_metric.update(output, target)

            if (batch_idx % 100 == 0) and (args.rank == 0) and args.wandb:
                wandb.log({"loss": loss.item()})
                wandb.log({"examples": wandb.Image(data[0].detach().cpu(), caption=f"Label: {target[0].item()}")})

            if args.rank == 0:
                train_loader.set_description(f'loss: {loss.item():.4f}')

        
        scheduler.step()
        train_acc = train_acc_metric.compute()

        # ----------------- Validation -----------------
        model.eval()
        val_acc_metric.reset()
        if args.rank == 0:
            pbar = tqdm(test_loader, colour='green', desc=f'Validation Epoch {epoch}')
        else:
            pbar = test_loader

        with torch.no_grad():
            for data, target in pbar:
                data = data.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)
                with autocast_ctx:
                    output = model(data)
                val_acc_metric.update(output, target)

        val_acc = val_acc_metric.compute()

        # ----------------- Epoch result & checkpoint -----------------
        if args.rank == 0:
            print(f"[Epoch {epoch}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            ckpt_path = os.path.join(args.ckpt_dir, f'checkpoint_{epoch}.pth')
            save_checkpoint_fsdp(model, optimizer, epoch, scheduler, ckpt_path, rank=args.rank)

    # ====================== Finish ===============================
    dist.barrier()
    init_end_event.record()
    if args.wandb and args.rank == 0:
        wandb.finish()
    if args.rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000} sec")
    dist.destroy_process_group()