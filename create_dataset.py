import os
import csv
import sys
import json
import pickle
import argparse
from tqdm.auto import tqdm
from data_module.preprocessing import ChatMessage

csv.field_size_limit(sys.maxsize)


def open_messages(
    users_messages_fpath,
    good_messages_fpath,
    bad_messages_fpath,
    false_positive_messages_fpath=None
):
    good_messages = open_chat_logs(good_messages_fpath)
    bad_messages = open_chat_logs(bad_messages_fpath)
    false_positive_messages = open_chat_logs(false_positive_messages_fpath)

    with open(users_messages_fpath, 'r', encoding='utf-8') as json_file:
        users_messages = json.load(json_file)
        users_messages = {
            username: [ChatMessage(*message) for message in messages]
            for username, messages in users_messages.items()
        }

    return users_messages, good_messages, bad_messages, false_positive_messages


def open_chat_logs(fpath):
    chat_messages = []

    with open(fpath, 'r', newline='') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')

        for message in tsv_reader:
            chat_message = ChatMessage(*message)
            text = chat_message.text
            if (
                not text.startswith('DELETEDMESSAGE') and
                not text == 'TIMEOUT' and
                not text == 'BAN'
            ):
                chat_messages.append(chat_message)

    return chat_messages


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

        if kwargs.get('include_channel'):
            datum['channel'] = channel

        if kwargs.get('include_category'):
            datum['category'] = message.category

        if kwargs.get('num_previous_user_messages'):
            datum['previous_user_messages'] = previous_messages(
                users_messages[message.username], message, kwargs['num_previous_user_messages']
            )

        if kwargs.get('num_context_messages'):
            datum['context'] = context_messages(
                chat_messages, message, kwargs['num_context_messages']
            )

        data.append((datum, label))

    return data


def previous_messages(user_messages, chat_message, k=5):
    index = user_messages.index(chat_message)
    previous_texts = [
        message.text
        for message in user_messages[index-k: index]
    ]
    previous_texts = '\n'.join(previous_texts)

    return previous_texts


def context_messages(chat_messages, chat_message, k=2):
    index = chat_messages.index(chat_message)
    context_texts = [
        f'{message.username}: {message.text}'
        for message in chat_messages[index-k: index+k]
    ]
    context_texts = '\n'.join(context_texts)

    return context_texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_directory', type=str)
    parser.add_argument('--processed_data_directory', type=str)
    parser.add_argument('--output_file_path', type=str)
    parser.add_argument('--include_channel', action='store_true')
    parser.add_argument('--include_category', action='store_true')
    parser.add_argument('--num_previous_user_messages', type=int)
    parser.add_argument('--num_context_messages', type=int)

    args = parser.parse_args()
    data = []
    raw_data_directory = args.raw_data_directory
    processed_data_directory = args.processed_data_directory
    for file_name in tqdm(os.listdir(raw_data_directory)):
        channel, _ = os.path.splitext(file_name)
        raw_data_fpath = os.path.join(raw_data_directory, file_name)
        users_messages_fpath = os.path.join(processed_data_directory, f'{channel}_users_messages.json')
        good_messages_fpath = os.path.join(processed_data_directory, f'{channel}_good_messages.tsv')
        bad_messages_fpath = os.path.join(processed_data_directory, f'{channel}_bad_messages.tsv')
        false_positive_messages_fpath = os.path.join(processed_data_directory, f'{channel}_false_positive_messages.tsv')

        try:
            chat_messages = open_chat_logs(raw_data_fpath)
            users_messages, good_messages, bad_messages, false_positive_messages = open_messages(
                users_messages_fpath,
                good_messages_fpath,
                bad_messages_fpath,
                false_positive_messages_fpath
            )

            good_dataset = create_dataset(
                channel, chat_messages, users_messages, good_messages, 0, **vars(args)
            )
            bad_dataset = create_dataset(
                channel, chat_messages, users_messages, bad_messages, 1, **vars(args)
            )
            false_positive_dataset = create_dataset(
                channel, chat_messages, users_messages, false_positive_messages, 0, **vars(args)
            )

            data.extend(good_dataset)
            data.extend(bad_dataset)
            data.extend(false_positive_dataset)
        except:
            print(f'Error with {channel}')

    with open(args.output_file_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
