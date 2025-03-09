import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import concurrent.futures

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image

from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from model_irse import Backbone

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, IPFRAttnProcessor2_0 as IPFRAttnProcessor
else:
    from attention_processor import IPAttnProcessor, AttnProcessor, IPFRAttnProcessor


####### Dataset ##########
def load_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            caption = json_data.get('caption', '')
            image_path = os.path.join(os.path.dirname(json_path), os.path.basename(json_path).replace('.json', '.jpg'))
            return {'image_path': image_path, 'caption': caption}
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {json_path}")
    except UnicodeDecodeError:
        print(f"Error reading file due to Unicode decode error: {json_path}")
    return None


def get_ffhq_dataset(num_processors=4, root_path='/mnt/d/ffhq_wild_files'):

    all_items = os.listdir(root_path)
    subdirectories = [os.path.join(root_path, item) for item in all_items if os.path.isdir(os.path.join(root_path, item))]

    data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processors) as executor:
        future_to_json = {executor.submit(load_json, os.path.join(root_path, folder, file_name)): file_name
            for folder in subdirectories
            for file_name in os.listdir(os.path.join(root_path, folder))
            if file_name.endswith('.json')}
        for future in concurrent.futures.as_completed(future_to_json):
            result = future.result()
            if result:
                data.append(result)


    dataset = Dataset.from_dict({'image': [item['image_path'] for item in data], 'caption': [item['caption'] for item in data]}).cast_column('image', datasets.Image())

    return dataset

class FFHQDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, transforms, size=512, 
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05):
        super().__init__()
        
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        
        self.clip_image_processor = CLIPImageProcessor()
        self.transforms = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.hf_dataset)

    def fr_image_preprocess(self, raw_image):
        """
        PIL 
        """
        raw_image = raw_image.resize((256, 256), Image.BILINEAR)
        img = self.to_tensor(raw_image)
        img = img * 2 - 1
        img = torch.unsqueeze(img, 0)
        return img

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        text = item["caption"]
        image = item["image"]
        raw_face_image = item["face_image"]

        clip_image = self.clip_image_processor(images=image, return_tensors="pt").pixel_values
        face_image = self.fr_image_preprocess(raw_face_image)

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image": image,
            "face_image": face_image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed
        }

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    face_images = torch.stack([example["face_image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {"images": images,
            "face_images": face_images,
           "text_input_ids": text_input_ids,
           "clip_images": clip_images,
           "drop_image_embeds": drop_image_embeds
           }


# Models
class ImageProjModel(torch.nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class FRImageProjModel(torch.nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, fr_embeddings_dim=512, fr_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.fr_extra_context_tokens = fr_extra_context_tokens
        
        self.proj = torch.nn.Linear(fr_embeddings_dim, self.fr_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, fr_embeds):
        """
        fr_embeds : [B, 512]
        clip_extra_context_tokens : [B, fr_extra_context_tokens, cross_attention_dim] 
        """
        embeds = fr_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.fr_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class FRIPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, face_proj_model=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        if face_proj_model is not None:
            self.face_proj_model = face_proj_model 

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, face_embeds=None):
        ip_tokens = self.image_proj_model(image_embeds)
        if face_embeds is not None:
            fr_tokens = self.face_proj_model(face_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens, fr_tokens], dim=1)
        # Predict the noise residual and compute loss
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred
        
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--face_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to Face image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/purestorage/project/tyk/tmp",
        help=("The cache directory")
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(    
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", cache_dir=args.cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=args.cache_dir)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", cache_dir=args.cache_dir)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", cache_dir=args.cache_dir)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path, cache_dir=args.cache_dir)
    face_encoder = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
    face_encoder.load_state_dict(torch.load(args.face_encoder_path))
    face_encoder.eval()
    
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    face_encoder.requires_grad_(False)

    #ip-adapter
    image_proj_model = FRImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    )

    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            # 기존 weight로 로드 시켜줌
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPFRAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                fr_embeds_dim=512)
            attn_procs[name].load_state_dict(weights)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    ip_adapter = FRIPAdapter(unet, image_proj_model, adapter_modules)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    face_encoder.to(accelerator.device, dtype=weight_dtype)

    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(), ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    hf_dataset = get_ffhq_dataset(num_processors=16)
    train_dataset = FFHQDataset(hf_dataset, tokenizer=tokenizer, size=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                    face_embeds = face_encoder(batch["face_images"].to(accelerator.device, dtype=weight_dtype))
                image_embeds_ = []
                face_embeds_ = []
                for image_embed, face_embed, drop_image_embed in zip(image_embeds, face_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.apppend(torch.zeros_like(image_embed))
                        face_embeds_.append(torch.zeros_like(face_embed))
                    else:
                        image_embeds_.append(image_embed)
                        face_embeds_.append(face_embed)
                image_embeds = torch.stack(image_embeds_)
                face_embeds = torch.stack(face_embeds_)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds, face_embeds)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
                    
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

            begin = time.perf_counter()
            
if __name__ == "__main__":
    main()
