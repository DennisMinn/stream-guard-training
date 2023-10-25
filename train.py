import argparse
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from data_module.twitch import TwitchDataModule
from model.twitch import TwitchModel
from logger.twitch import TwitchLogger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--num_warmup_steps', type=float)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--weight_decay', type=float)

    args = parser.parse_args()

    # Data Setup
    data_module = TwitchDataModule(
        file_path=args.input_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=1
    )
    data_module.setup()

    # Model Setup
    num_training_steps = int(args.num_epochs * data_module.train_size / args.batch_size)
    num_warmup_steps = int(num_training_steps * args.num_warmup_steps)

    model = TwitchModel(
        model_name=args.model_name,
        num_training_steps=num_training_steps,
        learning_rate=args.learning_rate,
        num_warmup_steps=num_warmup_steps,
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    # Logger Setup
    logger = TwitchLogger(**vars(args))
    wandb_logger = WandbLogger()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Training
    trainer = L.Trainer(
        logger=wandb_logger,
        enable_progress_bar=False,
        enable_checkpointing=False,
        max_epochs=args.num_epochs,
        callbacks=[logger, lr_monitor]
    )

    trainer.fit(model, data_module)
