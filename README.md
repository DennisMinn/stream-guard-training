# Overview:
This repository prepares the data gathered by [stream-scraper-bot](https://github.com/DennisMinn/stream-scraper-bot) to train the Stream Guard Bot.  Stream Guard Bot is a deep-learning moderation chatbot finely tuned using [OpenAI](https://platform.openai.com/docs/guides/fine-tuning).

# Preprocessing Script: `preprocessing.py`

This script is designed for preprocessing chat logs to filter out undesirable messages and organize them into separate categories of "good" and "bad" messages. The script uses a set of rules to clean and classify messages, with the final output stored in tab-separated values (TSV) files.

## Usage

```bash
python preprocessing.py --input_path <path_to_input_file> --output_directory <output_directory_path>
```

# Training Script: `train.py`

This script is designed for training a Twitch model using the Lightning framework. It incorporates key functionalities for data setup, model configuration, and training, along with logging and monitoring capabilities.

## Usage

```bash
python train.py --input_path <path_to_input_data> --model_name <model_name> --batch_size <batch_size> --num_epochs <num_epochs> --learning_rate <learning_rate> --num_warmup_steps <num_warmup_steps> --eps <epsilon_value> --weight_decay <weight_decay_value>
```

- `--input_path`: Path to the input data for training.
- `--model_name`: Name of the Twitch model to be trained.
- `--batch_size`: Batch size for training.
- `--num_epochs`: Number of training epochs.
- `--learning_rate`: Learning rate for optimization.
- `--num_warmup_steps`: Number of warm-up steps for learning rate scheduling.
- `--eps`: Epsilon value for optimizer.
- `--weight_decay`: Weight decay value for regularization.

Ensure you have the following dependencies installed to run the training script:

- [Python](https://www.python.org/) (version 3.6 or later)
- [Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) (2.1.0 or later) 
- [Weights & Biases](https://wandb.ai/site) (account)
