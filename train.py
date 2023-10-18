import argparse
import lightning as L
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

    model = TwitchModel(
        model_name=args.model_name,
        num_training_steps=num_training_steps,
        learning_rate=args.learning_rate
    )

    # Logger Setup
    logger = TwitchLogger(**vars(args))

    # Training
    trainer = L.Trainer(
        logger=False,
        enable_progress_bar=False,
        max_epochs=args.num_epochs,
        callbacks=[logger]
    )

    trainer.fit(model, data_module)
