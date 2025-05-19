import os
import time
import argparse

import torch
import torch.distributed as dist


from utils import get_lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)


    # ============================ Distributed Setting ============================
    init_distributed_mode(cfg)

    # =========================== Training Setting ===============================
    fix_random_seeds(cfg.seed+cfg.rank)

    cfg.tokens_per_iter = cfg.gradient_accumulation_steps * cfg.rank * cfg.batch_size * cfg.block_size

    set_torch_backends_ampare()

    # ============================ Dataset =========================================
    data_dir = os.path.join('data', cfg.dataset)
    def get_batch(split):
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))

        x = torch.stack(
            [torch.from_numpy((data[i:i+cfg.block_size]).astype(np.int64)) for i in ix]
        )

        y = torch.stack(
            [torch.from_numpy((data[i+1:i+1+cfg.block_size]).astype(np.int64)) for i in ix]
        )

        if cfg.device == 'cuda':
            x, y = x.pin_memory().to(cfg.device, non_blocking=True), y.pin_memory().to(cfg.device, non_blocking=True)
        else:
            x, y = x.to(cfg.device), y.to(cfg.device)

        return x, y

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_data = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # ============================ Models =========================================
    if cfg.init_from == 'scratch':
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        
        cfg.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
        model = GPT(cfg)

    elif cfg.init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        ckpt = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt, map_location=cfg.device)
        checkpoint_model_args = checkpoint['model_args']

        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            cfg[k] = checkpoint_model_args[k]

        model = GPT(cfg)
        model.load_state_dict(checkpoint['model_state_dict'])

        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    elif cfg.init_from == 'gpt2':
        print(f"Initializing from OpenAI GPT-2 weights: {cfg.init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=cfg.dropout)
        model = GPT.from_pretrained(init_from, override_args)
        
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            cfg[k] = getattr(model.config, k)
        
    # crop down the model block size if desired, using model surgery
    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)

    model.to(cfg.device)

    if cfg.world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank
        )

    # ============================ Optimizer and Loss ======================================

    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        **extra_args
    )
    
    print(f"using fused AdamW: {use_fused}")


    # ============================ Train ==============================================

    iter_num = 0
    best_val_loss = 1e9

    X, Y = get_batch('train') # fetch the very first batch
    raw_model = model.module if cfg.world_size > 1 else model # unwrap DDP container if needed
    running_mfu = -1.0

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0 and cfg.is_master:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train_loss": losses['train'],
                    "val_loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100
                })

            if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': cfg,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(gradient_accumulation_steps):
            if cfg.world_size > 1:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

            with autocast_ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # clip the gradient
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer) # grad / scale (scale에 곱해진 상태이므로 원래대로 되돌림)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % cfg.log_interval == 0 and cfg.is_master:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    # ============================ Finish ==============================================
    if cfg.wandb and cfg.rank == 0:
        wandb.finish()
    if cfg.is_master:
        dist.destroy_process_group()