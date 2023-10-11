import os
import csv
import sys
import re
import json

from collections import namedtuple, defaultdict, Counter

csv.field_size_limit(sys.maxsize)

ChatMessage = namedtuple('ChatMessage', ['timestamp', 'category', 'username', 'text'])


def write_chat_logs(
    users_messages,
    good_messages,
    bad_messages,
    input_fpath,
    output_directory
):
    def _write_chat_logs(fpath, chat_messages):
        with open(fpath, 'w', newline='') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')

            for message in chat_messages:
                tsv_writer.writerow(message)

    file_name = os.path.basename(input_fpath)
    channel, _ = os.path.splitext(file_name)
    _write_chat_logs(f'{output_directory}/{channel}_good_messsages.tsv', good_messages)
    _write_chat_logs(f'{output_directory}/{channel}_bad_messsages.tsv', bad_messages)

    with open(f'{output_directory}/{channel}_users_messages.json', 'w') as json_file:
        json.dump(users_messages, json_file, indent=4)


def open_chat_logs(fpath):
    chat_messages = []

    with open(fpath, 'r', newline='') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')

        for message in tsv_reader:
            chat_message = ChatMessage(*message)
            chat_messages.append(chat_message)

    return chat_messages


def chat_log_to_messages(chat_messages):
    users_messages = defaultdict(list)
    good_messages = set()
    bad_messages = set()

    for message in chat_messages:
        username, text = message.username, message.text

        if (
            (text.startswith('DELETEDMESSAGE') or text == 'TIMEOUT' or text == 'BAN') and
            len(users_messages[username]) and
            users_messages[username][-1].text != text  # Check for spam
        ):
            bad_messages.add(users_messages[username][-1])
        else:
            good_messages.add(message)
            users_messages[username].append(message)

    good_messages = good_messages.difference(bad_messages)

    good_messages = [message for message in good_messages if clean_good_message(message)]
    bad_messages = [message for message in bad_messages if clean_bad_message(message)]

    return {
        'users_messages': users_messages,
        'good_messages': good_messages,
        'bad_messages': bad_messages
    }


def clean_good_message(message):
    username, text = message.username, message.text

    if username == 'streamelements' or username.endswith('bot'):
        return False

    # Different language
    if len(text) != len(text.encode()):
        return False

    return True


def clean_bad_message(message):
    text = message.text
    if text.startswith('!'):
        return False

    # Cap-lock
    if len(re.findall(r'[A-Z]', text)) > len(re.findall(r'[a-z]', text)):
        return False

    # Different language
    if len(text) != len(text.encode()):
        return False

    website_pattern = r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)'
    if bool(re.search(website_pattern, text, flags=re.IGNORECASE)):
        return False

    spam_pattern = r'(\S+(?:\s+\S+)*)(?:\s+\1){3,}'
    if bool(re.search(spam_pattern, text, flags=re.IGNORECASE)):
        return False

    return True


def clean_post_processing_messages(good_messages, bad_messages):
    if len(bad_messages):
        most_common_message = Counter([message.text.lower() for message in bad_messages]).most_common(1)
        most_common_message = most_common_message[0][0]
        bad_messages = [
            message for message in bad_messages
            if most_common_message not in message.text.lower()
        ]

    return good_messages, bad_messages
