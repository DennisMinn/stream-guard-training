import argparse
import random

from data_module.preprocessing import (
    open_chat_logs,
    chat_log_to_messages,
    clean_post_processing_messages,
    write_chat_logs
)


def clean_chat_logs(trainer, model, data_module):
    data_module.setup()
    bad_messages = data_module.bad_dataset.messages
    good_messages = random.sample(
        data_module.good_dataset.messages, len(bad_messages)
    )

    good_messages, bad_messages = clean_post_processing_messages(good_messages, bad_messages)

    return good_messages, bad_messages


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_directory', type=str)

    args = parser.parse_args()

    chat_messages = open_chat_logs(args.input_path)
    users_messages, good_messages, bad_messages = chat_log_to_messages(chat_messages).values()
    good_messages, bad_messages = clean_post_processing_messages(good_messages, bad_messages)
    write_chat_logs(
        users_messages, good_messages, bad_messages, args.input_path, args.output_directory
    )
