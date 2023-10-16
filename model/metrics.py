from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def accuracy(twitch_items):
    y_true = [item.label for item in twitch_items]
    y_pred = [item.prediction for item in twitch_items]
    return accuracy_score(y_true, y_pred)


def precision(twitch_items):
    y_true = [item.label for item in twitch_items]
    y_pred = [item.prediction for item in twitch_items]
    return precision_score(y_true, y_pred)


def recall(twitch_items):
    y_true = [item.label for item in twitch_items]
    y_pred = [item.prediction for item in twitch_items]
    return recall_score(y_true, y_pred)


def f1(twitch_items):
    y_true = [item.label for item in twitch_items]
    y_pred = [item.prediction for item in twitch_items]
    return f1_score(y_true, y_pred)
