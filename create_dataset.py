import os
import json
import argparse
from data_module.preprocessing import (
    open_chat_logs,
    ChatMessage
)


def open_messages(users_messages_fpath, good_messages_fpath, bad_messages_fpath):
    good_messages = open_chat_logs(good_messages_fpath)
    bad_messages = open_chat_logs(bad_messages_fpath)

    with open(users_messages_fpath, 'r', encoding='utf-8') as json_file:
        users_messages = json.load(json_file)
        users_messages = {
            username: [ChatMessage(*message) for message in messages]
            for username, messages in users_messages.items()
        }

    return users_messages, good_messages, bad_messages


def create_dataset(
    channel,
    chat_messages,
    users_messages,
    messages,
    label,
    **kwargs
):
    data = []
    for message in messages:
        datum = {'text': message.text}

        if kwargs.get('channel'):
            datum['channel'] = channel

        if kwargs.get('category'):
            datum['category'] = message.category

        if kwargs.get('previous_messages'):
            datum['previous_messages'] = previous_messages(
                users_messages, message, kwargs['previous_user_messages']
            )

        if kwargs.get('context_messages'):
            datum['context'] = context_messages(
                chat_messages, message, kwargs['context_messages']
            )

        data.append((datum, label))

    return data


def previous_messages(user_messages, chat_message, k=5):
    index = user_messages.index(chat_message)
    previous_texts = [
        f'{message.username}: {message.text}'
        for message in user_messages[index-k: index]
    ]
    previous_texts = '\n'.join(previous_messages)

    return previous_texts


def context_messages(chat_messages, chat_message, k=2):
    index = chat_messages.index(chat_message)
    context_texts = [
        f'{message.username}: {message.text}'
        for message in chat_messages[index-k: index+k]
    ]
    context_texts = '\n'.join(context_messages)

    return context_texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat_log_directory', type=str)
    parser.add_argument('--messages_directory', type=str)
    parser.add_argument('--output_fpath', type=str)
    parser.add_argument('--channel')
    parser.add_argument('--category')
    parser.add_argument('--previous_user_messages', type=int)
    parser.add_argument('--context_user_messages', type=int)

    args = parser.parse_args()
    data = []
    for chat_log_fpath in os.listdir(args.chat_log_directory):
        channel, _ = os.path.splitext(chat_log_fpath)

        chat_messages = open_chat_logs(chat_log_fpath)
        users_messages, good_messages, bad_messages = open_messages(
            f'{channel}_users_messages.json',
            f'{channel}_good_messages.tsv',
            f'{channel}_bad_messages.tsv'
        )

        good_dataset = create_dataset(
            channel, chat_messages, users_messages, good_messages, 0 **vars(args)
        )
        bad_dataset = create_dataset(
            channel, chat_messages, users_messages, bad_messages, 1, **vars(args)
        )
        data.extend(good_dataset)
        data.extend(bad_dataset)

    with open(args.output_fpath, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)

