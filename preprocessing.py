import argparse
import wandb

import lightning as L
from data_module.preprocessing import (
    PreprocessingDataModule,
    remove_common_message
)
from model.preprocessing import (
    PreprocessingModel,
    similar_message_indices,
    false_positive_message_indices
)


def clean_chat_logs(trainer, model, data_module):
    data_module.setup()
    outputs = trainer.predict(model, data_module)

    if outputs is None:
        return [], [], []

    good_outputs, bad_outputs = outputs

    good_embeddings = [batch_output['embeddings'] for batch_output in good_outputs]
    classifications = [batch_output['classifications'] for batch_output in good_outputs]
    bad_embeddings = [batch_output['embeddings'] for batch_output in bad_outputs]

    banned_usernames = set([message.username for message in data_module.bad_dataset])
    false_positive_messages = set([
        data_module.good_dataset[index]
        for index in false_positive_message_indices(classifications)
        if data_module.good_dataset[index].username not in banned_usernames
    ])

    similar_messages = set([
        data_module.good_dataset[index]
        for index in similar_message_indices(good_embeddings, bad_embeddings)
    ])

    good_messages = similar_messages
    bad_messages = remove_common_message(data_module.bad_dataset.messages)

    return good_messages, bad_messages, false_positive_messages


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_directory', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int)

    args = parser.parse_args()

    trainer = L.Trainer(
        logger=False,
        enable_progress_bar=False
    )
    model = PreprocessingModel(args.model_name)
    data_module = PreprocessingDataModule(
        file_path=args.input_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=8
    )

    wandb.init(project='Preprocessing', name=args.input_path.split('/')[-1])
    good_messages, bad_messages, false_positive_messages = clean_chat_logs(trainer, model, data_module)

    data_module.good_messages = good_messages
    data_module.bad_messages = bad_messages
    data_module.false_positive_messages = false_positive_messages

    data_module.export(args.output_directory)
    wandb.finish()
