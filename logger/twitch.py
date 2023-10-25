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
        wandb.define_metric('train/step')
        wandb.define_metric('train/*', step_metric='train/step')

        wandb.define_metric('validation/step')
        wandb.define_metric('validation/*', step_metric='validation/step')

        self.config = kwargs

        self.train = []
        self.train_loss = 0
        self.train_step = 0

        self.validation = []
        self.validation_loss = 0
        self.validation_step = 0

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

    def on_train_epoch_end(self, trainer, lightning_module):
        # Log Metrics
        metrics = {
            'train/loss': self.train_loss / trainer.datamodule.train_size,
            'train/accuracy': accuracy(self.train),
            'train/precision': precision(self.train),
            'train/recall': recall(self.train),
            'train/f1': f1(self.train),
            'train/step': self.train_step
        }

        wandb.log(metrics)

        # Reset metrics
        self.train = []
        self.train_loss = 0
        self.train_step += 1

    def on_validation_epoch_end(self, trainer, lightning_module):
        from dataclasses import asdict

        # Log data
        columns = list(asdict(self.validation[0]).keys())
        data = [
            list(asdict(item).values())
            for item in self.validation
        ]
        table = wandb.Table(data=data, columns=columns)

        # Log Metrics
        metrics = {
            'validation/loss': self.validation_loss / trainer.datamodule.val_size,
            'validation/accuracy': accuracy(self.validation),
            'validation/precision': precision(self.validation),
            'validation/recall': recall(self.validation),
            'validation/f1': f1(self.validation),
            'validation/data': table,
            'validation/step': self.validation_step
        }

        wandb.log(metrics)

        # Reset Metrics
        self.validation = []
        self.validation_loss = 0
        self.validation_step += 1

    def teardown(self, trainer, lightning_module, stage):
        wandb.finish()
