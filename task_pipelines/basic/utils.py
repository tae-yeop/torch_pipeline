import yaml
import random
import numpy as np
import torch
import os
import argparse
from omegaconf import OmegaConf
import wandb

import torch
import torch.nn as nn

def yaml_load(file_path: str):
    f = open(file_path, 'r')
    data = yaml.safe_load(f)
    f.close()

    return data



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


def setup_trainable_params(model:nn.Module, trainable_modules:Tuple[str] = ("attn1.to_q", "attn2.to_q", "attn_temp",)):
    model.requires_grad_(False)
    for name, module in model.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True



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