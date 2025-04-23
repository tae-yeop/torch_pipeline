import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, Module

import torch.nn.functional as F
import torch.optim as optim

from torch.cuda import amp

import torchvision
import torchvision.transforms as transforms

import torchmetrics
from torchmetrics.classification import MulticlassAccuracy

import os
import sys
import importlib
import argparse
import math
from tqdm.auto import tqdm

source =  os.path.join(os.getcwd(), '..') # /home/tyk/train-box/torch_basic/.. 을 추가함
if source not in sys.path:
    sys.path.append(source)

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exec_type, exc_value, traceback):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(decription='Single GPU Training')

    # Optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
    # Scheduler
    # parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
    parser.add_argument('--epochs', default=100, type=float, help='Training epochs')
    # Dataset
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
    # Dataloader
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    # Model
    parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
    parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
    parser.add_argument('--prenorm', action='store_true', help='Prenorm')
    # General
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16'])
    parser.add_argument('--check_out_dir', type=str, default='./checkpoint')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        if args.grayscale:
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
                transforms.Lambda(lambda x : x.view(1, 1024).to())
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: x.view(3, 1024).t())
            ])

        train_set = torchvision.datasets.CIFAR10(
            root='./data/cifar', train=True, download=True, transform=transform
        )
        train_set, _ = split_train_val(train_set, val_split=0.1)

        val_set = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=True, download=True, transform=transform)
        _, val_set = split_train_val(val_set, val_split=0.1)

        test_set = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=False, download=True, transform=transform)

        d_input = 3 if not args.grayscale else 1
        d_output = 10
        
    elif args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
    else:
        raise NotImplementedError


    train_sampler = None
    val_sampler = None
    test_sampler = None

    train_loader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=args.batch_size,
                                            shuffle=(train_sampler is None),
                                            pin_memory=True,
                                            num_workers=2,
                                            sampler=train_sampler,
                                            drop_last=True,
                                            persistent_workers=True)
    
    val_loader = torch.utils.data.DataLoader(val_set, 
                                         batch_size=args.batch_size,
                                         shuffle=False, 
                                         pin_memory=True,
                                         num_workers=2,
                                         sampler=val_sampler,
                                         drop_last=False,)
    
    test_loader = torch.utils.data.DataLoader(test_set, 
                                          batch_size=args.batch_size, 
                                          shuffle=False, 
                                          pin_memory=True,
                                          num_workers=2,
                                          sampler=test_sampler,
                                          drop_last=False,)
    
    # Model
    if args.model == 'resnet':
        # from model.resnet.res_models import Model
        model_cls = getattr(importlib.import_module('model.resnet.res_models'), "Model")
    elif args.model == 'vit':
        model_cls = getattr(importlib.import_module('model.transformer.vit'), "ViT")
            
    model = model_cls()

    # Optimizer
    all_parameters = list(model.parameters())

    # 특정 레이어에 _optim key라는게 있음 (보통은 없음)
    # 일반 레이어는 AdamW로 보낸다
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # 특정 레이어의 특정 파라마티의 attribute (_optim)에 optimizer 관련 정보 저장해줌
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]

    # 이 부분이 왜 굳이 있는지?
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]

    # list of dict를 looping하면서
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp} # weight_decay, lr을 넣음
        )

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'reducedlr':
        if args.patience is None:
            raise ValueError("You must specify patience.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.2)
    else:
        raise NotImplementedError


    # Print optimizer info
    # optim이 있는 것만 key를 얻음 (아까 별로도 설정했던 것들에 대해 프린팅)
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))


    # loss 사용
    if args.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']


    if args.precision == "fp16":
        ptdtype = torch.float16
    elif args.precision == "bf16":
        ptdtype = torch.bfloat16
    else:
        ptdtype = torch.float32

    # Underfitting을 방지
    scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == 'float16'))

    pbar = tqdm(range(start_epoch, args.epochs))
    for epoch in pbar:
        if epoch == 0:
            pbar.set_description('Epoch: %d' % (epoch))
        else:
            pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))
            
        # Train
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to('cuda')
            targets = targets.to('cuda')

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(device_type='cuda', dtype=ptdtype) if args.fp16 else dummy_context_mgr():
                output = model(inputs)
                loss = criterion(output, targets)

            scaler.scale(loss).backward() # if scaler is None, this is equivalent to loss.backward()
            scaler.unscale_(optimizer) # gradient norm 적용을 위해서 https://pytorch.org/docs/stable/amp.html
            
            # gradient norm
            if args.gradient_norm == True:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                is_overflow = math.isnan(grad_norm)


            scaler.step(optimizer) # if scaler is None, this is equivalent to optimizer.step()
            scaler.update() # if scaler is None, this is equivalent to None

        # Validation
        model.eval()
        pbar = tqdm(range(len(test_loader)), colour='green', desc='Validation Epoch')

        with torch.no_grad():
            for batch_idx, (inputs, targets) in pbar:
                inputs = inputs.to('cuda', pin_memory=True, non_blocking=True)
                targets = targets.to('cuda', pin_memory=True, non_blocking=True)
            
                output = model(inputs)
                loss = criterion(outputs, targets)


            if epoch % 10 == 0:
                if acc > best_acc:
                    state = {
                        'model': model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict' : scheduler.state_dict(),
                        'args': args,
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')

                    # 저장할 때 step이나 epoch을 체크포인트 이름에 넣어서 활용 가능
                    torch.save(state, os.path.join(args.check_out_dir, f'ckpt_{str(epoch).zfill(6)}.pt'))
                    best_acc = acc
                    # scheduler.step(val_acc)




        val_acc = test(model, val_loader)
        test(model, test_loader)
        scheduler.step()