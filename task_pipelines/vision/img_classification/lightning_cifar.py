import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything
from lightning.pytorch.strategies import DDPStrategy
from lightning import LightningModule, Trainer

import timm
import argparse
import importlib
import os
import wandb

def instantiate_from_config(config):
    """
    예: 
    config = {
        "target": "my_module.MyClass",
        "params": { ... }
    }
    -> get_obj_from_str(config["target"])(**config["params"])
    """

    if not "target" in config:
        if config in ["__is_first_stage__", "__is_unconditional__"]:
            return None
        raise KeyError("Expected key `target` to instantiate.")
    # model.traget이 되기 떄문에 cldm.cldm.ControlLDM가 됨
    # Model class : cldm.cldm.ControlLDM 여기에 params을 kwargs형태로 __init__에 넣는 꼴
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    """
    문자열("package.module.Class")을 실제 Python 객체로 변환
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# Model
class ViTLightningModel(LightningModule):
    def __init__(
            self, 
            model_name="vit_base_patch16_224", 
            pretrained=True, 
            num_classes=10, 
            lr=1e-4,
            weight_path=None
            
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=num_classes,
        )

        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            print(f"[load_model] missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")


        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        """
        loss를 리턴하는 메소드
        
        training_step에선 내부적으로 다음이 실행됨.

        # put model in train mode and enable gradient calculation
        model.train()
        torch.set_grad_enabled(True)

        for batch_idx, batch in enumerate(train_dataloader):
            loss = training_step(batch, batch_idx)

            # clear gradients
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update parameters
            optimizer.step()
        """
        x, y = batch # y=[batch]
        logits = self(x) # [batch, num_classes]
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean() # [True or False] -> [0. or 1.] -> mean along batch axis

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # [batch, num_classes]
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training config
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (per node).")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for distributed training.")
    parser.add_argument("--val_freq", type=float, default=1.0, help="Frequency of validation checks.")
    parser.add_argument("--deterministic", action="store_true", help="Set to True for deterministic training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # Model config
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224", help="timm model name.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes.")
    # Resume checkpoint
    parser.add_argument("--resume_ckpt_path", type=str, default=None, help="Path to resume checkpoint.")
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb_project", type=str, default="ai_service_model")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_key", type=str, default=None, help="Wandb API key")
    parser.add_argument("--wandb_host", type=str, default=None,)
    parser.add_argument("--wandb_run_name", type=str, default="test_1")
    parser.add_argument("--resume_ckpt_path", type=str, default=None,)
    args = parser.parse_args()
    
    if args.deterministic:
        seed_everything(args.seed, workers=True)

    # Wandb
    wandb_logger = None
    if args.use_wandb:
        wandb.login(key=args.wandb_key, host=args.wandb_host, force=True,)
        wandb_logger = WandbLogger(name='test_1', project='ai_service_model', log_model=True)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # imagenet mean and std
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_size = int(len(train_data) * 0.9)
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
    
    # Models
    if args.resume_ckpt_path:
        model = ViTLightningModel.load_from_checkpoint(args.resume_ckpt_path)
    else:
        model = ViTLightningModel(
            model_name="vit_base_patch16_224",
            num_classes=10,
            lr=1e-4,
        )

    
    # Strategy
    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)

    # Trainer
    trainer_cfg = dict(accelerator="gpu", 
                       gpus=args.gpus, 
                       precision=32, 
                       num_nodes=args.num_nodes, 
                       strategy=ddp,
                       logger=wandb_logger, 
                       max_epochs=args.max_epochs, 
                       val_check_interval=args.val_freq,
                       logger=wandb_logger if wandb_logger else True)
    
    if args.deterministic:
        trainer_cfg.update({'deterministic': True})


    trainer = Trainer(
        **trainer_cfg
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)

    print("Training and testing complete!")










    # Load for pure pytorch
    ckpt = torch.load(checkpoint)
    vit_weights = {k:v for k,v in ckpt["state_dict"].items() 
                        if k.startswith("vit.")}
    model.load_state_dict(vit_weights)
    model.eval()