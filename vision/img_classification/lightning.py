import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything
from lightning.pytorch.strategies import DDPStrategy

import timm
from lightning import LightningModule, Trainer
import argparse

class ViTLightningModel(LightningModule):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True, num_classes=10, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=num_classes,
        )

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
    parser.add_argument("--resume_ckpt_path", type=str, default=None,)
    args = parser.parse_args()

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
    

    if args.resume_ckpt_path:
        model = ViTLightningModel.load_from_checkpoint(args.resume_ckpt_path)
    else:
        model = ViTLightningModel(
            model_name="vit_base_patch16_224",
            num_classes=10,
            lr=1e-4,
        )

    wandb.login(key=wandb_key, host=wandb_host, force=True,)
    wandb_logger = WandbLogger(name='test_1', project='ai_service_model', log_model=True)

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)

    trainer_cfg = dict(accelerator="gpu", 
                       gpus=args.gpus, 
                       precision=32, 
                       num_nodes=args.num_nodes, 
                       strategy=ddp,
                       logger=wandb_logger, 
                       max_epochs=args.max_epochs, 
                       val_check_interval=args.val_freq)
    
    if args.deterministic:
        seed_everything(args.seed, workers=True)
        trainer_cfg.update({'deterministic': True})


    trainer = Trainer(
        **trainer_cfg
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)











    # Load for pure pytorch
    ckpt = torch.load(checkpoint)
    vit_weights = {k:v for k,v in ckpt["state_dict"].items() 
                        if k.startswith("vit.")}
    model.load_state_dict(vit_weights)
    model.eval()