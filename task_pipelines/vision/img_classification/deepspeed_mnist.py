"""
참조
https://github.com/microsoft/DeepSpeedExamples/blob/master/cifar/cifar10_deepspeed.py
https://github.com/CUBOX-Co-Ltd/cifar10_DDP_FSDP_DS/blob/master/deepspeed_cifar10.py
https://github.com/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/08_zero_redundancy_optimization.ipynb
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

import torch.distributed as dist

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as tvm

import numpy as np
import deepspeed
import argparse
import datetime
import torchmetrics


def hw_setup_ampare():
    """
    Ampere 아키텍처 (RTX30xx, A100, H100, ...)에서 
    TensorFloat-32 활용을 위해 설정합니다.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_ds_config(args):
    ds_config = {
        "train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 2000,
        "optimizer":{
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler":{
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr":0,
                "warmup_max_lr":0.001,
                "warmup_num_steps":1000,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": args.dtype == "bf16"}, # args에 따라서 True 걸리도록
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
        "activation_checkpointing":{
            "partition_activations": True,
            "cpu_checkpointing": True, # 매우 큰 activation 텐서는 CPU로 오프로딩함
            
        },
        
    }
    return ds_config

def get_date_of_run():
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def prepare_mnist(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if rank != 0:
        dist.barrier()

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if rank == 0:
        dist.barrier()

    # 분산 Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler  = DistributedSampler(test_dataset,  num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, sampler=test_sampler,  num_workers=4)

    return train_loader, test_loader


def prepare_cifar10(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # rank=0이 데이터 다운로드하고, barrier 이후 나머지가 접근하도록
    if rank != 0:
        dist.barrier()

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if rank == 0:
        dist.barrier()


    # deepspeed.initialize는 training dataset까진 자동으로 샘플러까지 처리해줌
    # 결국 test dataset이라도 결국 수동으로 하긴 해야함.
    # 분산 Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler  = DistributedSampler(test_dataset,  num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, sampler=test_sampler,  num_workers=4)
    
    return train_loader, test_loader, train_sampler


def get_model(args):
    if args.dataset == "mnist":
        model = tvm.resnet18(weights=None)
        # 원래 ResNet-18은 conv1 입력 채널 수가 3이므로, 1채널에 맞춰 수정
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, 10)  # MNIST = 10 classes
    elif args.data == 'cifar10':
        model = tvm.swin_v2_b(weights=None)  # 필요하면 pretrained=... 사용
        model.head = nn.Linear(1024, 10)
    else:
        raise ValueError("Invalid dataset. Please choose 'cifar10' or'mnist'")
    
    return model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--stage", type=int, default=1, help="ZeRO optimization stage")
    parser.add_argument("--dataset", type=str, default="mnist") # cifar10
    parser.add_argument("--precision", type=int, default=32) # 16, 32
    # -- deepspeed.launch 시 자동으로 추가되는 인자들 --
    parser.add_argument("--local_rank", type=int, default=0)

    # 만약에 json에 config를 넣어서 쓰고 각 config를 args로 조회하고 싶다면
    # 실행도 이렇게 해야함 python train.py --deepspeed --deepspeed_config deepspeed_config.json
    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # ======================
    # 1) 분산 초기화
    # ======================
    # DeepSpeed에서 init_distributed를 호출해 NCCL 환경을 초기화
    deepspeed.init_distributed(dist_backend="nccl")

    # ======================
    # 2) local_rank 설정, 해당 GPU에 맞춰 연산
    # ======================
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ======================
    # 3) Dataset
    # ======================
    if args.dataset == "mnist":
        train_loader, test_loader, train_sampler = prepare_mnist(args)
    elif args.dataset == "cifar10":
        train_loader, test_loader, train_sampler = prepare_cifar10(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    
    # ======================
    # 4) Model
    # ======================
    model = get_model(args)


    # ======================
    # 5) DeepSpeed 초기화
    # ======================
    # DeepSpeed Config 생성
    ds_config = get_ds_config(args)
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_dataloader=train_loader,
        config=ds_config
    )

    criterion = nn.CrossEntropyLoss()

    # ======================
    # 6) Metrics
    # ======================

    train_accuracy = torchmetrics.Accuracy(
        task="multiclass", 
        num_classes=10,
        dist_sync_on_step=True    # 분산 환경에서 step마다 통계 동기화
    ).to(model_engine.local_rank)


    test_accuracy = torchmetrics.Accuracy(
        task="multiclass",
        num_classes=10,
        dist_sync_on_step=True
    ).to(model_engine.local_rank)
  

    # ======================
    # 6) Loop
    # ======================
    for epoch in range(args.epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"=== Epoch [{epoch+1}/{args.epochs}] ===")
            data_iter = tqdm(train_loader)
        else:
            data_iter = train_loader

        for step, (images, labels) in enumerate(data_iter):
            images = images.to(model_engine.local_rank)
            labels = labels.to(model_engine.local_rank)

            outputs = model_engine(images)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            # --- torchmetrics update ---
            preds = outputs.argmax(dim=1)
            train_accuracy.update(preds, labels)

            if step % 10 == 0 and rank == 0:
                acc_val = train_accuracy.compute().item()
                data_iter.set_description(f"Loss: {loss.item():.4f} | Accuracy: {acc_val*100:.2f}%")


        model_engine.eval()
        test_accuracy.reset() 
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(model_engine.local_rank)
                labels = labels.to(model_engine.local_rank)
                outputs = model_engine(images)
                preds = outputs.argmax(dim=1)
                test_accuracy.update(preds, labels)

        epoch_test_acc = test_accuracy.compute().item()
        if rank == 0:
            print(f"[Epoch {epoch+1}] Test Accuracy: {epoch_test_acc*100:.2f}%\n")

    dist.barrier()
    if rank == 0:
        print("Training Complete!")