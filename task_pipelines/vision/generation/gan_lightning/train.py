import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything



if __name__ == '__main__':
    args = get_args_with_config()


    # -----------------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------------
    trainer = pl.Trainer(**trainer_cfg)
    trainer.fit(coach_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)