import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from model_irse import Backbone
import torch.nn.functional as F

from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, IPFRAttnProcessor2_0 as IPFRAttnProcessor
else:
    from attention_processor import IPAttnProcessor, AttnProcessor, IPFRAttnProcessor



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
    

class IPAdapterFR:
    def __init__(self, sd_pipe, image_encoder_path, face_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path, cache_dir=cache_dir).to(self.device, dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()
        
        # load face encoder
        self.face_encoder = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.face_encoder.load_state_dict(torch.load(face_encoder_path))
        self.face_encoder.eval()

        # image proj model
        self.image_proj_model, self.fr_proj_model = self.init_proj()

        self.load_ip_adapter()
        
    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)

        face_proj_model =  FRImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model, face_proj_model 

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
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
                                                    fr_embeds_dim=512, scale1=1.0, scale2=1.0, image_num_tokens=4,
                                                    fr_num_tokens=4).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.face_proj_model.load_state_dict(state_dict["fa"])

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, face_embeds=None):
        ip_tokens = self.image_proj_model(image_embeds)
        if face_embeds is not None:
            fr_tokens = self.face_proj_model(face_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens, fr_tokens], dim=1)
        # Predict the noise residual and compute loss
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))

        return image_prompt_embeds, uncond_image_prompt_embeds
        
    def set_scale(self, scale1, scale2):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale1
            if isinstance(attn_processor, IPFRAttnProcessor):
                attn_processor.scale1 = scale1
                attn_processor.scale2 = scale2
    
    def generate(
        self,
        pil_image,
        face_image,
        prompt=None,
        negative_prompt=None,
        scale1=1.0,
        scale2=1.0,
        num_samples=4,
        seed=-1,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale1, scale2)

        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts