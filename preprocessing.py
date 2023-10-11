import argparse

import lightning as L
from data_module.preprocessing import (
    PreprocessingDataModule,
    clean_post_processing_messages
)
from model.preprocessing import (
    PreprocessingModel,
    similar_message_indices,
    false_positive_message_indices
)


def clean_chat_logs(trainer, model, data_module):
    data_module.setup()
    good_dataloader, bad_dataloader = data_module.predict_dataloader()
    outputs = trainer.predict(model, [good_dataloader, bad_dataloader])

    if outputs and outputs[0]:
        good_embeddings = [batch_output['embeddings'] for batch_output in outputs[0]]
        classifications = [batch_output['classifications'] for batch_output in outputs[0]]
    else:
        good_embeddings = []
        classifications = []

    if outputs and outputs[1]:
        bad_embeddings = [batch_output['embeddings'] for batch_output in outputs[1]]
    else:
        bad_embeddings = []

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

    good_messages = similar_messages.union(false_positive_messages)
    bad_messages = data_module.bad_dataset.messages
    good_messages, bad_messages = clean_post_processing_messages(good_messages, bad_messages)

    return good_messages, bad_messages


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

    data_module.good_messages, data_module.bad_messages = clean_chat_logs(trainer, model, data_module)
    try:
        data_module.export(args.output_directory)
    except:
        print(f'Error with {args.input_path}')
