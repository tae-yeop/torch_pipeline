import os
import sys
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# timm에서 Vision Transformer 계열 모델 불러오기
import timm
# timm.models.vision_transformer 내부 모듈
from timm.models.vision_transformer import Block  # timm의 Transformer block class

# ---- [FSDP 관련: PyTorch 2.x 최신 API] ----
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
    summon_full_params,
    optim_state_dict,
    load_optim_state_dict,
    BackwardPrefetch, # 백워드 패스 최적화를 조금 더 하기 위해 있는 옵션
)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap, # FSDP wrap을 수동으로 적용할 때 사용
    wrap,
)

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


from torch.optim.lr_scheduler import StepLR
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy
from functools import partial
from tqdm import tqdm

def resolve_root_node_address(nodes: str) -> str:
    """(참고 함수) SLURM 등의 노드리스트 처리용"""
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
    """간단한 PyTorch 분산 초기화"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"]) 
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 단일 GPU 실행 시(테스트 용도)
        print('No RANK/WORLD_SIZE found, running on single GPU.')
        args.rank, args.local_rank, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    # PyTorch DDP/FSDP init
    os.environ["MASTER_ADDR"] = get_main_address()
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)
    dist.barrier()
    print(f"[Rank {args.rank}] world_size = {args.world_size}, local_rank = {args.local_rank}")

def fix_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# -----------------------------------------
# FSDP 체크포인트 저장/로드 (PyTorch 2.x 권장)
# -----------------------------------------
def save_ckpt_fsdp(model, optimizer, scheduler, epoch, ckpt_path, rank=0):
    if rank != 0:
        return

    # summon_full_params: full state_dict 얻기
    with summon_full_params(model):
        full_sd = model.state_dict()
    full_osd = optim_state_dict(model, optimizer)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": full_sd,
        "optimizer_state_dict": full_osd,
        "scheduler_state_dict": scheduler.state_dict(),
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"[Rank {rank}] Saved checkpoint to {ckpt_path}")

def load_ckpt_fsdp(model, optimizer, scheduler, ckpt_path, rank=0):
    import torch.distributed as dist
    dist.barrier()  # 동기화
    if rank == 0:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        ckpt = None
    dist.barrier()

    with summon_full_params(model):
        if rank == 0:
            model.load_state_dict(ckpt["model_state_dict"])
    dist.barrier()

    if rank == 0:
        optim_sd = ckpt["optimizer_state_dict"]
        scheduler_sd = ckpt["scheduler_state_dict"]
        loaded_epoch = ckpt["epoch"] + 1
    else:
        optim_sd = None
        scheduler_sd = None
        loaded_epoch = 0

    load_optim_state_dict(model, optimizer, optim_sd)
    dist.barrier()

    if rank == 0:
        scheduler.load_state_dict(scheduler_sd)
        print(f"[Rank 0] Resume from epoch {loaded_epoch}")
    return loaded_epoch

# -----------------------------------------
# 간단한 학습 루프
# -----------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch, rank=0):
    model.train()
    train_acc_metric = MulticlassAccuracy(num_classes=10).to(device)

    loader_iter = loader
    if rank == 0:
        loader_iter = tqdm(loader, desc=f"[Train Epoch {epoch}]", leave=False)
    for batch_idx, (images, labels) in enumerate(loader_iter):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # autocast (bf16, fp16, or fp32 => FSDP MixedPrecision와 맞춤)
        with torch.autocast(device_type='cuda', dtype=next(model.parameters()).dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_acc_metric.update(outputs, labels)
        if (batch_idx % 50 == 0) and (rank == 0):
            loader_iter.set_description(f"loss={loss.item():.4f}")

    # 전체 rank accuracy
    acc = train_acc_metric.compute()
    return acc.item()

def evaluate(model, loader, criterion, device, rank=0):
    model.eval()
    val_acc_metric = MulticlassAccuracy(num_classes=10).to(device)
    val_loss_sum = 0.0
    total_count = 0

    loader_iter = loader
    if rank == 0:
        loader_iter = tqdm(loader, desc="[Eval]", leave=False, colour='green')
    with torch.no_grad():
        for images, labels in loader_iter:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=next(model.parameters()).dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_acc_metric.update(outputs, labels)
            bs = images.size(0)
            val_loss_sum += loss.item() * bs
            total_count += bs
    val_acc = val_acc_metric.compute()
    val_loss = val_loss_sum / total_count if total_count > 0 else 0
    return val_acc.item(), val_loss

# -----------------------------------------
# 메인 실행 함수
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--precision", type=str, default="bf16", help="fp16|bf16|fp32")
    parser.add_argument("--resume_ckpt", type=str, default="")

    parser.add_argument("--use_enable_wrap", action="store_true",
                        help="If set, use enable_wrap/wrap manually; else pass auto_wrap_policy to FSDP constructor.")
    parser.add_argument("--backward_prefetch", type=str, default="none",
                        help="Can be none|BACKWARD_PRE|BACKWARD_POST for FSDP backward_prefetch param.")
    args = parser.parse_args()

    init_distributed_mode(args)
    fix_random_seeds(42 + args.rank)

    # MNIST -> 3채널, 224x224로 변경
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    transform_test = transform_train  # 동일하게 사용

    # MNIST Dataset
    if args.rank == 0:
        datasets.MNIST(root="./data", train=True, download=True)
        datasets.MNIST(root="./data", train=False, download=True)
    dist.barrier()

    train_dataset = datasets.MNIST(root="./data", train=True, download=False, transform=transform_train)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=False, transform=transform_test)

    # 분산 샘플러
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    test_sampler  = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                              num_workers=4, pin_memory=True)
    

    # ============ timm Transformer 모델 (ViT 등) ============
    # 예) 'vit_small_patch16_224', pretrained=False, num_classes=10
    base_model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=10,
        in_chans=3
    ).to(args.device)

    # ============ FSDP: auto_wrap_policy = transformer_auto_wrap_policy ============
    # timm의 vit는 내부에 vision_transformer.Block 클래스를 사용하므로, 그것을 auto_wrap 대상에 설정
    # 해당 클래스(Transformer 레이어) 단위로 FSDP wrap이 자동 적용
    #  SwinTransformer 계열 모델은 SwinTransformerBlock 등을 지정
    auto_wrap_p = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})

    # MixedPrecision 설정
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
    
    # backward_prefetch 설정
    backward_prefetch_opt = None
    if args.backward_prefetch.upper() == "BACKWARD_PRE":
        backward_prefetch_opt = BackwardPrefetch.BACKWARD_PRE
    elif args.backward_prefetch.upper() == "BACKWARD_POST":
        backward_prefetch_opt = BackwardPrefetch.BACKWARD_POST
    else:
        backward_prefetch_opt = None  # none

    # -----------------------------------------------------------
    # [방법1] 그냥 FSDP 생성자 인자로 auto_wrap_policy 넣기
    # [방법2] enable_wrap + wrap를 통해 수동 래핑
    # -----------------------------------------------------------
    if not args.use_enable_wrap:
        # [방법1] 전역 auto wrapping
        fsdp_config = {
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "auto_wrap_policy": auto_wrap_p,
            "mixed_precision": mp_policy,
            "cpu_offload": CPUOffload(offload_params=False),
            "device_id": args.local_rank,
        }
        if backward_prefetch_opt is not None:
            fsdp_config["backward_prefetch"] = backward_prefetch_opt
        model = FSDP(base_model, **fsdp_config)
    else:
        # [방법2] 부분적으로 wrap
        # 예: base_model.blocks (transformer block 리스트)를 직접 wrap
        with enable_wrap(
            wrapper_cls=FSDP,
            auto_wrap_policy=auto_wrap_p,
            mixed_precision=mp_policy,
            cpu_offload=CPUOffload(offload_params=False),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=args.local_rank,
            backward_prefetch=backward_prefetch_opt,
        ):
            # 예) timm ViT는 base_model.blocks가 nn.ModuleList로 Transformer Block을 가짐
            for i in range(len(base_model.blocks)):
                base_model.blocks[i] = wrap(base_model.blocks[i])

            # 머리나 패치 임베딩, 기타는 wrap하지 않고 그대로 둠
            # 필요하다면 base_model.patch_embed, base_model.head 등도 wrap 가능

        # 모델을 FSDP로 wrap
        # 마지막에 전체 모델을 다시 한 번 FSDP로 감싸도 됨(파라미터 flatten, shard)
        # auto_wrap_policy가 적용된 부분들은 중첩 FSDP 구조가 만들어짐
        model = FSDP(base_model, device_id=args.local_rank)

    # 옵티마이저, 스케줄러
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    # fp16에서만 GradScaler 활성화
    scaler = ShardedGradScaler(enabled=(args.precision == "fp16"))

    # Resume
    start_epoch = 0
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        start_epoch = load_ckpt_fsdp(model, optimizer, scheduler, args.resume_ckpt, rank=args.rank)

    # ----------------- 학습 루프 -----------------
    best_val_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, args.device, epoch, args.rank)
        val_acc, val_loss = evaluate(model, test_loader, criterion, args.device, args.rank)
        scheduler.step()

        if args.rank == 0:
            print(f"[Epoch {epoch}]  train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")

            # 체크포인트 예시
            ckpt_path = f"checkpoint_epoch{epoch}.pth"
            save_ckpt_fsdp(model, optimizer, scheduler, epoch, ckpt_path, rank=args.rank)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best val_acc: {best_val_acc:.4f}")

    dist.barrier()
    if args.rank == 0:
        print("Training done. Best val_acc =", best_val_acc)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()