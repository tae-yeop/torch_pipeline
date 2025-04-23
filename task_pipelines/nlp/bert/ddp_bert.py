import os
import sys
import re
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda import amp

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from functools import partial

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def resolve_root_node_address(nodes: str) -> str:
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
        args.rank = int(os.environ['RANK'])
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



def tokenize_fn(examples, tokenizer):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--precision", type=str, default="fp32")
    args = parser.parse_args()

    init_distributed_mode(args)

    # Model
    model_name = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir='./tmp')
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3, cache_dir='./tmp').cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


    # Load dataset
    raw_datasets = load_dataset("multi_nli", split="train")
    partial_fn = partial(tokenize_fn, tokenizer=tokenizer)
    tokenized_dataset = raw_datasets.map(
        partial_fn,
        batched=True,  # 여러 샘플을 한 번에 처리
        remove_columns=["premise", "hypothesis"]  # 기존 텍스트 컬럼은 제거(옵션)
    )

    # label 컬럼이 'label'일 경우 rename (HF 모델 호환)
    # tokenized_dataset = tokenized_dataset.rename_column("label", "labels")  # 필요 시
        
    # PyTorch 텐서로 변환 : input_ids, attention_mask, label(또는 labels) 등만 남김
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]  # or "labels"
    )


    train_sampler = DistributedSampler(
        tokenized_dataset,
        rank=args.rank,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        collate_fn=data_collator,
        num_workers=4, 
        pin_memory=True
    )


    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
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

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()

    for epoch in range(args.epochs):
        model.train()

        if train_sampler:
            train_sampler.set_epoch(epoch)

        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["label"].to(args.device)  # 또는 batch["labels"]


            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if idx % 10 == 0 and args.rank == 0:
                print(f"step:{idx}, loss:{loss.item()}")


        scheduler.step()

        
    dist.barrier()
    init_end_event.record()

    if args.rank==0:
        torch.save(model.state_dict(), 'model.pt')
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
    dist.destroy_process_group()