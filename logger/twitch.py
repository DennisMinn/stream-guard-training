import wandb
from lightning import Callback
from model.metrics import accuracy, precision, recall, f1


class TwitchLogger(Callback):
    def __init__(self, **kwargs):
        from datetime import datetime
        super().__init__()
        wandb.init(
            name=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            config=kwargs
        )

        self.config = kwargs

        self.train = []
        self.validation = []
        self.train_loss = 0
        self.validation_loss = 0

    def on_train_batch_end(
        self,
        trainer,
        lightning_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self.train += outputs['twitch_items']
        self.train_loss += outputs['loss']

    def on_validation_batch_end(
        self,
        trainer,
        lightning_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0
    ):
        self.validation += outputs['twitch_items']
        self.validation_loss += outputs['loss']

    def on_train_end(self, trainer, lightning_module):
        # Log Metrics
        wandb.log({'train/loss': self.train_loss / trainer.datamodule.train_size})
        wandb.log({'train/accuracy': accuracy(self.train)})
        wandb.log({'train/precision': precision(self.train)})
        wandb.log({'train/recall': recall(self.train)})
        wandb.log({'train/f1': f1(self.train)})

        # Reset metrics
        self.train = []
        self.train_loss = 0

    def on_validation_end(self, trainer, lightning_module):
        from dataclasses import asdict
        # Log Metrics
        wandb.log({'validation/loss': self.validation_loss / trainer.datamodule.val_size})
        wandb.log({'validation/accuracy': accuracy(self.validation)})
        wandb.log({'validation/precision': precision(self.validation)})
        wandb.log({'validation/recall': recall(self.validation)})
        wandb.log({'validation/f1': f1(self.validation)})

        # Log Data
        columns = list(asdict(self.validation[0]).keys())
        data = [
            list(asdict(item).values())
            for item in self.validation
        ]
        table = wandb.Table(data=data, columns=columns)
        wandb.log({'validation/data': table})

        # Reset Metrics
        self.validation = []
        self.validation_loss = 0

    def teardown(self, trainer, lightning_module, stage):
        wandb.finish()
