
import random
import numpy as np
import torch

import yaml
import os
import importlib
import argparse
from omegaconf import OmegaConf
import wandb

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

import inspect
from inspect import isfunction
from functools import wraps

# Python Utils
def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# Configs
def yaml_load(file_path: str):
    f = open(file_path, 'r')
    data = yaml.safe_load(f)
    f.close()

    return data

def yaml_omegaconf(file_path: str):
    # filt_path : [yaml]
    # Retuns : indentation한건 dict of dict로 들어감
    config = OmegaConf.load(file_path)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    return args


def setup_wandb(wandb_info):
    wandb.login(key=wandb_info.get('key', 'local-d20a4c3fd6cffd419ca148decace4cb95004b226'), host=wandb_info.get('host','http://211.168.94.228:8080'), force=True,)
    wandb_logger = WandbLogger(name='test_1', project='ai_service_model', log_model=True)
    return wandb_logger


def get_args_with_config():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Example with YAML Config')
    parser.add_argument('--config', type=str, default='configs/cfg_example.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    assert args.config is not None
    args = OmegaConf.load(args.config)
    OmegaConf.set_struct(args, False)
    return args


# Dataset Preparation
def get_directory(root):
    """
    디렉토리와 파일 리스트 순회하기
    """
    root = next(os.walk(root))[0]
    한칸아래_디렉토리_리스트 = next(os.walk(root))[1]
    한칸아래_파일_리스트 = next(os.walk(root))[2]
    return 한칸아래_디렉토리_리스트, 한칸아래_파일_리스트

def get_directory_only(root):
    """
    파일은 제외하고 디렉토리만 가져오고 싶을 때.
    """
    return [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]


def generate_file_paths(directory):
    """
    사용예시
    directory = '/purestorage/datasets/laion_face_extracted/split_00001'
    for file_path in generate_file_paths(directory):
        print(file_path)
    """
    try:
        for foldername, _, filenames in os.walk(directory):
            for filename in filenames:
                yield os.path.join(foldername, filename)
    except Exception as e:
        print(f"An error occurred: {e}")

def get_data_path_list():
    allowed_extensions = [".jpg", ".png"]
    dataset_folder = '/purestorage/project/tyk/9_Animation/Dataset/Kaggle/data/train/color'
    out_dir = '/purestorage/project/tyk/9_Animation/Dataset/Kaggle/data/'

    only_paths = [str(path) for path in Path(dataset_folder).rglob("*") if path.suffix.lower() in allowed_extensions]

from torch.utils.data import Subset, random_split

def split_train_valid(dataset, val_indices = [1, 2, 3, 4, 5, 6, 7, 8]):
    all_indices = list(range(len(dataset)))
    train_indices = [idx for idx in all_indices if idx not in val_indices]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset

def split_train_valid_ratio(dataset, ratio=0.95):
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


# Dataset feeding conversion
def conversion(device, *items):
    return (data.to(device, pin_memory=True, non_blocking=True) for data in items)





# ============================ Pytorch Setup ========================================
def set_seed(seed):
    """
    reproducibility
    https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L31 참조
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# ============================ Pytorch Model ========================================
def create_model(config_path):
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    # 모델에 대한 dict를 넘겨줌 (model.target, model.params,...)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    # model.traget이 되기 떄문에 cldm.cldm.ControlLDM가 됨
    # Model class : cldm.cldm.ControlLDM 여기에 params을 kwargs형태로 __init__에 넣는 꼴
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def torch_dfs(model : nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


def save_checkpoint(model, path, module):
    """
    특정 모듈만 저장
    """
    mm_state_dict = OrderedDict()
    state_dict = model.state_dict()

    for key in state_dict:
        if module in key:
            mm_state_dict[key] = state_dict[key]
    torch.save(mm_state_dict, path)


def load_model_checkpoint(model, ckpt):
    """
    https://github.com/Doubiiu/DynamiCrafter
    파이토치 라이트닝으로 저장한 경우 state_dict가 있는 것을 처리해줌
    """
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, device):
    """
    safetensors 파일까지 처리하는 코드

    load_file(path) :  가장 간단하고 PyTorch 스타일로 state_dict 바로 얻고 싶을 때
    safe_open(path)	: 큰 모델에서 일부만 로드하거나, 메모리 최적화할 때
    load(data): 메모리에서 바로 로딩할 때 (예: 서버에서 받은 바이트 데이터)
    """

    _, extension = os.path.splitext(ckpt_path)

    if extension.lower() == ".safetensors":
        try:
            import safetensors.torch
        except ImportError:
            raise ImportError("Please install safetensors to load .safetensors checkpoints.")
        state_dict = safetensors.torch.load_file(ckpt_path, device=device)
    else:
        loaded = torch.load(ckpt_path, map_location=device)
        state_dict = get_state_dict(loaded)

    return state_dict


def setup_trainable_params(model:nn.Module, trainable_modules:Tuple[str] = ("attn1.to_q", "attn2.to_q", "attn_temp",)):
    model.requires_grad_(False)
    for name, module in model.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params

# ============================ Optimizers  ========================================
def wrap_kwargs(f):
    sig = inspect.signature(f)
    # Check if f already has kwargs
    has_kwargs = any([
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    ])
    if has_kwargs:
        @wraps(f)
        def f_kwargs(*args, **kwargs):
            y = f(*args, **kwargs)
            if isinstance(y, tuple) and isinstance(y[-1], dict):
                return y
            else:
                return y, {}
    else:
        param_kwargs = inspect.Parameter("kwargs", kind=inspect.Parameter.VAR_KEYWORD)
        sig_kwargs = inspect.Signature(parameters=list(sig.parameters.values())+[param_kwargs])
        @wraps(f)
        def f_kwargs(*args, **kwargs):
            bound = sig_kwargs.bind(*args, **kwargs)
            if "kwargs" in bound.arguments:
                kwargs = bound.arguments.pop("kwargs")
            else:
                kwargs = {}
            y = f(**bound.arguments)
            if isinstance(y, tuple) and isinstance(y[-1], dict):
                return *y[:-1], {**y[-1], **kwargs}
            else:
                return y, kwargs
    return f_kwargs

def discard_kwargs(f):
    if f is None: return None
    f_kwargs = wrap_kwargs(f)
    @wraps(f)
    def f_(*args, **kwargs):
        return f_kwargs(*args, **kwargs)[0]
    return f_

    
# ============================ Training  ========================================
def toggle_grad(model, on, freeze_layers=-1):
    '''
    https://github.com/VITA-Group/Ultra-Data-Efficient-GAN-Training/blob/main/BigGAN%20and%20DiffAugGAN/utils/misc.py
    '''
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        num_blocks = len(model.module.in_dims)
    else:
        num_blocks = len(model.is_dims)

    assert freeze_layers < num_blocks,"can't not freeze the {fl}th block > total {nb} blocks.".format(fl=freeze_layers, nb=num_blocks)
    if freeze_layers == -1:
        for name, param in model.named_parameters():
            param.requires_grad = on
    else:
        for name, param in model.named_parameters():
            param.requires_grad = on
        for layer in range(freeze_layers):
            block = "blocks.{layer}".format(layer=layer)
            if block in name:
                param.requires_grad = False

def set_dropout(model, new_p):
    """
    Dynamic droptout : iteration이 돌때마다 dropout 값을 바꿀 수 있게끔
    """
    for idx, m in enumerate(model.named_modules()):
        path = m[0]
        component = m[1]
        if isinstance(component, nn.Dropout):
            component.p = new_p

def adjust_lr(optimizer, lr):
    """
    Learning rate scheduling  from scratch
    """
    for group in optimizer:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def get_prev_step(args):
    """
    만약에 체크포인트 파일이름에 step수가 표기되어 있을시 사용
    https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py
    """
    if args.ckpt is not None:
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass


# dummy mixed precision : fp16 안켰을 때 이용
# 그냥 amp.autocast(enabled=...), enabled 사용하는게 편할듯
# https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/e95bcd46372573581ae8b34c083e65bd5e4e0e9e/src/worker.py#L10
class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exec_type, exc_value, traceback):
        return False
    
    


import torch
from torch.cuda.amp import autocast, GradScaler
# Multiple Model Multiple Optimizer
def train_multi_models(
    models,             # 모델들을 리스트로 받음 [model0, model1, ...]
    optimizers,         # 옵티마이저들을 리스트로 받음 [optimizer0, optimizer1, ...]
    loss_fns,           # 필요한 경우 loss 함수들도 리스트나 하나로 받음
    data_loader,        # 학습용 DataLoader
    epochs,             # 학습 횟수
    scaler,
    device='cuda'
):
    scaler = GradScaler()
    
    # 모델을 device로 이동
    for model in models:
        model.to(device)

    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # 배치를 device로 이동
            inputs, targets = inputs.to(device), targets.to(device)

            # 각각의 옵티마이저 zero_grad()
            for optimizer in optimizers:
                optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                # 모델 개수만큼 순전파
                outputs = [model(inputs) for model in models]
                
                # 예시) 모델0의 출력(output0)과 모델1의 출력(output1)을 조합하여 loss0, loss1 계산
                # 질문에서 언급하신 로직: 
                #   loss0 = loss_fn(2 * output0 + 3 * output1, target)
                #   loss1 = loss_fn(3 * output0 - 5 * output1, target)
                # 확장성이 필요한 경우, outputs를 받아서 어떻게 조합할지 따로 함수화 가능
                output0, output1 = outputs  # 모델이 2개라고 가정

                loss0 = loss_fns(2 * output0 + 3 * output1, targets)
                # retain_graph=True 로 인해 첫 backward 후에도 그래프 유지
                # (두 번째 backward 시 그래프를 그대로 사용해야 하는 경우)
                loss1 = loss_fns(3 * output0 - 5 * output1, targets)

            # 첫 backward 시 retain_graph=True
            scaler.scale(loss0).backward(retain_graph=True)
            scaler.scale(loss1).backward()

            # 필요에 따라 옵티마이저별 unscale()로 gradient 검증/수정 가능
            scaler.unscale_(optimizers[0])  # 꼭 특정 옵티마이저만 unscale할 필요는 없음

            # 모든 옵티마이저 step 진행
            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {batch_idx}] loss0: {loss0.item():.4f} loss1: {loss1.item():.4f}")



# Image Processing
def pil_rgb_convert(image):
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    return image

def _list_image_files_recursively(data_dir):
    file_list = []
    for home, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('gt.jpg'):
                file_list.append(os.path.join(home, filename))
    file_list.sort()
    return file_list

# Combination
from itertools import product
def get_combination(list1, list2):
    result = list(product(list1, list2))
    print(result)


# =====================
# numpy arrary mapper
# =====================
import numpy as np
from itertools import product
def save_data_mapper():
    list1 = ['A', 'B', 'C']
    list2 = [1,2,3,]
    result = list(product(list1, list2))
    # [('A', 1), ('A', 2), ('A', 3), ('B', 1), ('B', 2), ('B', 3), ('C', 1), ('C', 2), ('C', 3)]
    print(result)

    mapper = {}
    mapper['test'] = result
    file_array = np.array(result, dtype=object)
    np.save('file_array.npy', mapper)

save_data_mapper()

def load_data_mapper(npy_path='file_array.npy'):
    data_mapper = np.load(npy_path, allow_pickle=True)
    print(type(data_mapper))
    print(data_mapper)

    print(sum([len(v) for k, v in data_mapper.item().items()]))

    print(data_mapper.item()['test'][0])

load_data_mapper()
# {'test': [('A', 1), ('A', 2), ('A', 3), ('B', 1), ('B', 2), ('B', 3), ('C', 1), ('C', 2), ('C', 3)]}


import numpy as np
from itertools import product
def efficient_data_mapper():
    # 큰 데이터셋 생성
    list1 = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    list2 = list(range(1000))

    # NumPy 사용
    combinations = np.array(list(product(list1, list2)))
    indices = np.arange(len(combinations))

    # 메모리 효율적인 방법으로 딕셔너리 생성
    indexed_dict = dict(zip(indices, map(tuple, combinations)))

    print(f"Dictionary size: {len(indexed_dict)}")
    print(f"First few items: {list(indexed_dict.items())[:5]}")
    print(indexed_dict[0])



# ============================ logging  ========================================
from PIL import Image, ImageDraw, ImageFont
def log_txt_as_img(wh, xc, size=10):
    """
    텍스트 문자열 리스트를 이미지로 시각화하여 Tensor로 반환하는 함수.

    Args:
        wh (tuple): 이미지의 (width, height)
        xc (list): 문자열 리스트 (batch size 만큼의 캡션들)
        size (int): 폰트 크기

    Returns:
        torch.Tensor: (B, 3, H, W) 형태의 텍스트 이미지 텐서, [-1, 1] 정규화됨
    """
    b = len(xc)  # 배치 사이즈
    txts = list()

    for bi in range(b):
        # 흰 배경의 빈 이미지 생성
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)

        # 폰트 설정 (DejaVuSans.ttf는 별도로 설치되어 있어야 함)
        font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)

        # 한 줄에 표시할 최대 문자 수 추정 (너비 기준)
        nc = int(40 * (wh[0] / 256))

        # 긴 문장을 줄바꿈하여 여러 줄로 만듦
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            # 텍스트를 이미지에 그림
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            # 인코딩 문제로 그릴 수 없는 경우 경고 출력 후 스킵
            print("Cant encode string for logging. Skipping.")

        # 이미지 → numpy array → (C, H, W) 로 전환 후 [-1, 1] 정규화
        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)

    # 배치 텐서로 변환
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


import datetime
def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run
