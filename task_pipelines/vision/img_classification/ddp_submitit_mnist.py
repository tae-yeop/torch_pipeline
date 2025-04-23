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

import submitit  # Submitit 라이브러리

# ------------------------------------------------------------------------
# 1) Model & Utilities
# ------------------------------------------------------------------------
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
    Ampare architecture : 30xx, A100, H100, ...
    TensorFloat-32를 허용해 연산속도를 높임.
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
    """
    rank=0 프로세스가 아닌 경우 print를 무시하기 위한 헬퍼
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def resolve_root_node_address(nodes: str) -> str:
    """
    SLURM_NODELIST에서 첫 번째 노드 이름을 파싱해 오는 함수.
    예:
      'host0 host1 host3' -> 'host0'
      'host0,host1,host3' -> 'host0'
      'host[5-9]' -> 'host5'
    """
    nodes = re.sub(r"\[(.*?)[,-].*\]", "\\1", nodes)  # Take the first node of every node range
    nodes = re.sub(r"\[(.*?)\]", "\\1", nodes)        # handle special case where node range is single number
    return nodes.split(" ")[0].split(",")[0]


def get_main_address():
    """
    MASTER_ADDR를 설정(혹은 가져오기)하는 함수
    """
    root_node = os.environ.get("MASTER_ADDR")
    if root_node is None:
        nodelist = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        root_node = resolve_root_node_address(nodelist)
        os.environ["MASTER_ADDR"] = root_node
    return root_node


def init_distributed_mode(args):
    """
    기존 코드의 분산 초기화 로직.
    """
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
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    is_master = (args.rank == 0)
    dist.barrier()

    setup_for_distributed(is_master)
    print('| distributed init (rank {}): {}'.format(args.rank, os.environ['MASTER_ADDR']), flush=True)


# ------------------------------------------------------------------------
# 2) Training Logic as a Function
# ------------------------------------------------------------------------
def train_main(args):
    # 1) 분산 초기화
    init_distributed_mode(args)

    # 2) 기타 설정
    if args.rank == 0 and args.wandb:
        wandb.login(key=args.wandb_key, force=True)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    fix_random_seeds(args.seed + args.rank)
    set_torch_backends_ampare()

    # 3) Dataset 준비
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dist.get_rank() != 0:
        dist.barrier()

    train_dataset_tmp = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_tmp, [50000, 10000])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if dist.get_rank() == 0:
        dist.barrier()

    # DistributedSampler
    if args.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, rank=args.rank, num_replicas=args.world_size, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, rank=args.rank, num_replicas=args.world_size, shuffle=False)
    else:
        train_sampler, test_sampler = None, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )

    # 4) Model
    model = SimpleCNN().cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )

    # 5) Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1)

    # 6) Precision / AMP
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

    scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16))

    # 7) (Optional) Resume
    start_epoch = 0
    if args.resume:
        ckpt_filepath = os.path.join(args.ckpt_dir, args.ckpt_filename)
        map_location = lambda storage, loc: storage.cuda(args.local_rank)
        checkpoint = torch.load(ckpt_filepath, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}", flush=True)

    # 8) Training Loop
    train_acc_metric = MulticlassAccuracy(num_classes=10).to(args.local_rank)
    val_acc_metric = MulticlassAccuracy(num_classes=10).to(args.local_rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()

    for epoch in range(start_epoch, args.epochs):
        # --- Train ---
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_acc_metric.reset()

        # rank=0에서만 tqdm 표시
        if args.rank == 0:
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        else:
            loader_iter = train_loader

        for batch_idx, (data, target) in enumerate(loader_iter):
            data = data.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_acc_metric.update(output, target)

            if (batch_idx % 100 == 0) and (args.rank == 0) and args.wandb:
                wandb.log({"loss": loss.item()})
                images = wandb.Image(data[0].detach().cpu(), caption=f"Label: {target[0].item()}")
                wandb.log({'examples': images})

            if args.rank == 0:
                loader_iter.set_description(f"loss: {loss.item():.4f}")

        scheduler.step()

        train_acc = train_acc_metric.compute()

        # --- Validation ---
        model.eval()
        val_acc_metric.reset()

        if args.rank == 0:
            pbar = tqdm(range(len(test_loader)), colour='green', desc='Validation Epoch')
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to('cuda', non_blocking=True)
                target = target.to('cuda', non_blocking=True)
                with autocast_ctx:
                    output = model(data)
                    val_loss = criterion(output, target)

                val_acc_metric.update(output, target)

                if args.rank == 0:
                    pbar.update(1)

        if args.rank == 0:
            pbar.close()

        val_acc = val_acc_metric.compute()

        # --- Save Checkpoint (rank=0 only) ---
        if args.rank == 0:
            print(f"[Epoch {epoch}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # for single-GPU inference later
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, f'checkpoint_{epoch}.pth')

    # 9) Finish
    dist.barrier()
    init_end_event.record()

    if args.wandb and args.rank == 0:
        wandb.finish()

    if args.rank == 0:
        elapsed_time = init_start_event.elapsed_time(init_end_event) / 1000
        print(f"Training done! Elapsed time: {elapsed_time:.2f} sec")

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    python train_submitit.py --wandb --wandb_key=<your_key> --epochs=5
    이렇게 하면 슬럼이 아닌 로컬에서도 동작함
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_filename", type=str, default="checkpoint.pth")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mnist_cnn_pytorch")
    parser.add_argument("--wandb_entity", type=str, default="ty-kim")
    parser.add_argument("--wandb_key", type=str, default="")
    parser.add_argument("--local_rank", type=int, default=0, help="Used internally for distributed mode")

    args = parser.parse_args()


    # 1) Submitit AutoExecutor
    executor = submitit.AutoExecutor(
        folder="submitit_logs_mnist",
        slurm_max_num_timeout=20
    )

    # 2) Executor 파라미터 설정
    executor.update_parameters(
        name="mnist_submitit_job",
        timeout_min=60,        # 1시간
        slurm_partition="gpu", # 실제 Slurm 파티션
        gpus_per_node=1,       # 노드당 GPU 개수
        tasks_per_node=1,      # 일반적으로 dist. training 시 node당 1개의 Python 프로세스
        cpus_per_task=4,       # CPU 쓰레드 (데이터 로드 등)
    )

    # 3) 잡 제출 -> train_main(args) 호출
    job = executor.submit(train_main, args)


    # 4) 결과 기다리기
    _ = job.result()  # 여기서는 train_main이 None 리턴
