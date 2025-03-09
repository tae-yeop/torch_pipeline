import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import get_model
from utils import yaml_load, set_seed, set_deterministic, get_args
from metric.metrics import MyAccuracy



def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def set_torch_backends(args):
    # Ampare architecture 30xx, a100, h100,..
    if torch.cuda.get_device_capability(0) >= (8, 0):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
    if args.inference : torch.set_grad_enabled(False)


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
        args.is_master = True
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
    args.is_master = args.rank == 0
    dist.barrier()

    setup_for_distributed(args.rank == 0)
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    
    # 사용한 config 저장
    if args.is_master:
        log_path = f'./{args.log_dir}/{args.expname}'
        os.makedirs(log_path, exist_ok=True)
        save_args_to_yaml(args, f'{log_path}/config.yaml')

def get_logger(args):
    try:
        import wandb
        wandb_avail = True
    except ImportError:
        wandb_avail = False

    if args.wandb and wandb_avail:
        wandb.login(key=args.wandb_key, force=True)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    # 추후에 기본 로거까지 추가해야함
    # logger.info(f'{global_rank}, {local_rank}, {world_size}, {device}, {seed}')
    else:
        logger = logging.getLogger('TEST')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)05s %(message)s \n\t--- %(filename)s line: %(lineno)d in %(funcName)s", '%Y-%m-%d %H:%M:%S')

        # 터미널 용 핸들러
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # 파일 저장용 핸들러
        file_handler = logging.FileHandler(f'{log_path}/{expname}.log', mode=file_log_mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 시작 메시지
        start_message = f"\n\n{'=' * 50}\nSession Start: {expname} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 50}"
        logger.info(start_message)

        logger.log_eval = MethodType(log_eval, logger)

    return logger
    return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mnist_cnn_pytorch")
    parser.add_argument("--wandb_entity", type=str, default="ty-kim")
    parser.add_argument("--wandb_key", type=str)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--expname", type=str, default="exp")
    args = parser.parse_args()
    return args



def get_args_with_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='path to config file')
    args = parser.parse_args()
    assert args.config is not None

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    return args

def save_args_to_yaml(args, filename='saved_config.yaml'):
    args_dict = vars(args)
    with open(filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)


def log_eval(self, idx, loss, acc):
    self.info(f'{idx} iteration | loss : {loss} | acc : {acc}')


def get_dataset(args):
    assert args.train_batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'
    trsf = get_transform(trans_type='UHDM_train', train_img_size=args.train_img_size)

import torch
import inspect

try:
    import bitsandbytes as bnb
    adam8bit_class = bnb.optim.Adam8bit
except ImportError:
    adam8bit_class = None
    # pass, raise ImportErro

try:
    import prodigyopt
    prodigy_class = prodigyopt.Prodigy
except ImportError:
    prodigy_class = None

optimizer_dict = {'adam': torch.optim.Adam, 'adam8bit': adam8bit_class, 'adamw': torch.optim.AdamW, 'prodigy': prodigy_class}

def prepare_optimizer_params(models, learning_rates):
    params_to_optimizer = []
    for model, lr in zip(models, learning_rates):
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        model_parameters_with_lr = {'params': model_parameters, 'lr':lr}
        params_to_optimizer.append(model_parameters_with_lr)
    return params_to_optimizer

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Argument Setting
    # -----------------------------------------------------------------------------
    args = get_args()

    cfg = yaml_load(args.config)


    # -----------------------------------------------------------------------------
    # Distributed Setting
    # -----------------------------------------------------------------------------
    init_distributed_mode(args)

    # -----------------------------------------------------------------------------
    # Basic Setting
    # -----------------------------------------------------------------------------
    seed = args.random_seed
    if seed is not None:
        set_random_seeds(seed) 

    set_torch_backends(args)
    
    set_seed(cfg.training.seed)
    set_deterministic()

    # -----------------------------------------------------------------------------
    # loggder
    # -----------------------------------------------------------------------------
    logger = get_logger(args)


    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    train_set, test_set = get_dataset(args)
    train_loader, test_loader, train_sampler = get_dataloader(train_set, test_set, args)

    # -----------------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------------
    model = get_model(args.model_name)


    # -----------------------------------------------------------------------------
    # Optimization
    # -----------------------------------------------------------------------------
    train_accuracy = MyAccuracy(dist_sync_on_step=True).to(model_engine.local_rank)
    test_accuracy  = MyAccuracy(dist_sync_on_step=True).to(model_engine.local_rank)

    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()

    for epoch in range(cfg.training.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        train_accuracy.reset()


        train_loader = tqdm(train_loader, desc = f'Epoch {epoch}', leave=False) if args.is_master else train_loader


        for step, (images, labels) in enumerate(train_loader):
            ...

            train_accuracy.update(outputs, labels)

            if step % 10 == 0 and rank == 0:
                # 현재까지 누적된 정확도 계산
                curr_acc = train_accuracy.compute()
                train_loader.set_description(f"Loss: {loss.item():.4f} | TrainAcc: {curr_acc*100:.2f}%")

        
        # Epoch 끝나고 한 번 더 최종 train accuracy
        final_train_acc = train_accuracy.compute()
        if rank == 0:
            print(f"[Epoch {epoch+1}] Final Train Accuracy: {final_train_acc*100:.2f}%")

        model_engine.eval()
        test_accuracy.reset()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(model_engine.local_rank)
                labels = labels.to(model_engine.local_rank)
                preds = model_engine(images)
                test_accuracy.update(preds, labels)

        test_acc_val = test_accuracy.compute()
        if rank == 0:
            print(f"[Epoch {epoch+1}] Test Accuracy: {test_acc_val*100:.2f}%")
