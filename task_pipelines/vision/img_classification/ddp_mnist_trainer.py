import os
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# ------------------------- Simple CNN -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ============================================================
#   Coach: 모델, 데이터셋, 학습/평가 로직을 묶은 클래스
# ============================================================
class Coach:
    def __init__(self, cfg):
        """
        cfg (dict-like): 필요한 설정이 들어있는 객체(명령줄 인자 등)
            - cfg.device, cfg.rank, cfg.batch_size, cfg.epochs ...
        """
        self.cfg = cfg
        self.device = cfg.device
        self.rank = cfg.rank

        # (1) build model, optimizer, scaler, etc.
        self.build_model_and_optimizer()

        # (2) resume or not
        if cfg.resume:
            self.resume_from_ckpt()

        # (3) dataloader
        self.train_loader, self.val_loader = self.build_dataloader()

        # (4) AMP scaler
        # fp16인 경우에만 스케일러를 켠다는 예시
        if cfg.precision == 'fp16':
            self.scaler = GradScaler(enabled=True)
        else:
            self.scaler = GradScaler(enabled=False)

        # (5) loss function
        self.criterion = nn.CrossEntropyLoss()

    def build_model_and_optimizer(self):
        """
        모델, optimizer, scheduler 세팅
        """
        # 간단히 CNN 예시
        model = SimpleCNN().to(self.device)

        # DistributedDataParallel (DDP) 래핑
        # (cfg.world_size > 1) 인 경우만 DDP를 감싸도록 가정
        if self.cfg.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.cfg.local_rank],
                output_device=self.cfg.local_rank,
                find_unused_parameters=False
            )

        self.model = model

        # Optimizer, Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

    def build_dataloader(self):
        """
        MNIST dataloader 생성 (train/val 나눠서 예시)
        """
        # transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 다운로드는 rank=0에서만 진행
        if self.rank == 0:
            datasets.MNIST(root="./data", train=True, download=True)
            datasets.MNIST(root="./data", train=False, download=True)
        dist.barrier()  # 모든 rank가 다운로드가 끝날 때까지 대기

        full_train = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
        train_set, val_set = torch.utils.data.random_split(full_train, [50000, 10000])
        test_set = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

        # 분산 sampler
        if self.cfg.world_size > 1:
            train_sampler = DistributedSampler(train_set, num_replicas=self.cfg.world_size, rank=self.cfg.rank, shuffle=True)
            val_sampler = DistributedSampler(val_set, num_replicas=self.cfg.world_size, rank=self.cfg.rank, shuffle=False)
        else:
            train_sampler, val_sampler = None, None

        train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size, sampler=train_sampler,
                                  num_workers=4, pin_memory=True, shuffle=(train_sampler is None))
        val_loader = DataLoader(val_set, batch_size=self.cfg.batch_size, sampler=val_sampler,
                                num_workers=4, pin_memory=True, shuffle=False)

        return train_loader, val_loader

    def resume_from_ckpt(self):
        """
        체크포인트에서 로드하는 로직(예시)
        """
        ckpt_path = os.path.join(self.cfg.ckpt_dir, self.cfg.ckpt_filename)
        if not os.path.isfile(ckpt_path):
            if self.rank == 0:
                print(f"No checkpoint found at {ckpt_path}. Starting from scratch.")
            return

        map_location = lambda storage, loc: storage.cuda(self.cfg.local_rank)
        checkpoint = torch.load(ckpt_path, map_location=map_location)

        # DDP는 .module로 감싸있을 수 있으니 유의
        if self.cfg.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1

        if self.rank == 0:
            print(f"[Resume] Loaded checkpoint from {ckpt_path}, start_epoch={self.start_epoch}")
    # end resume_from_ckpt

    def save_checkpoint(self, epoch):
        """
        rank=0에서만 체크포인트 저장
        """
        if self.rank != 0:
            return
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.ckpt_filename}_epoch{epoch}.pth")

        if self.cfg.world_size > 1:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(ckpt, ckpt_path)
        print(f"[Rank 0] Saved checkpoint to {ckpt_path}")

    def train_loop(self):
        """
        전체 학습 루프
        """
        num_epochs = self.cfg.epochs
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch)
            self.validation(epoch)
            self.scheduler.step()
            self.save_checkpoint(epoch)

    def train_one_epoch(self, epoch):
        self.model.train()
        if isinstance(self.train_loader.sampler, DistributedSampler):
            # 분산 sampler라면 epoch마다 set_epoch
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_samples = 0
        correct = 0

        loader_iter = self.train_loader
        if self.rank == 0:
            loader_iter = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for data, target in loader_iter:
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(self.cfg.precision in ['fp16','bf16'])):
                output = self.model(data)
                loss = self.criterion(output, target)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Metrics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        # epoch stats (전체 rank 합산 가능하면 all_reduce)
        if self.rank == 0:
            epoch_loss = total_loss / total_samples
            epoch_acc = correct / total_samples
            print(f"[Train][Epoch {epoch}] loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

    def validation(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        loader_iter = self.val_loader
        if self.rank == 0:
            loader_iter = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

        with torch.no_grad():
            for data, target in loader_iter:
                data, target = data.to(self.device), target.to(self.device)
                with autocast(enabled=(self.cfg.precision in ['fp16','bf16'])):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        if self.rank == 0:
            val_loss = total_loss / total_samples
            val_acc = correct / total_samples
            print(f"[Val][Epoch {epoch}] loss={val_loss:.4f}, acc={val_acc:.4f}")

    def criterion(self, outputs, labels):
        """
        기본 CrossEntropyLoss
        """
        return self.criterion_fn(outputs, labels)

    @property
    def criterion_fn(self):
        if not hasattr(self, '_criterion'):
            self._criterion = nn.CrossEntropyLoss()
        return self._criterion

# ============================================================
#   Trainer: Coach를 생성하고, DDP init/seed 등을 관리
# ============================================================
class Trainer:
    def __init__(self, args):
        """
        args: argparse.Namespace (혹은 dict) 형태
        """
        self.args = args
        self.setup_dist()
        self.setup_seed()

        # Coach에 넘겨줄 cfg
        self.cfg = args  # 단순히 그대로 전달
        self.cfg.device = self.device
        self.cfg.rank = self.global_rank
        self.cfg.local_rank = self.local_rank
        self.cfg.world_size = self.world_size

        # Coach 초기화
        self.coach = Coach(self.cfg)

    def setup_dist(self):
        # 분산 초기화
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            # single GPU mode
            self.global_rank = 0
            self.world_size = 1
            self.local_rank = 0
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'

        dist.init_process_group(backend='nccl', world_size=self.world_size, rank=self.global_rank)
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)

    def setup_seed(self):
        seed = getattr(self.args, 'seed', 12345)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train(self):
        """
        코치가 준비된 상태에서 전체 학습을 수행
        """
        self.coach.train_loop()

    def close(self):
        dist.barrier()
        dist.destroy_process_group()

# ============================================================
#   MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--ckpt_filename", type=str, default="checkpoint")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32","fp16","bf16"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
    if trainer.global_rank == 0:
        print("Training complete.")
    trainer.close()

if __name__ == "__main__":
    main()
