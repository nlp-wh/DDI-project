from keras import backend as K
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def calculate_metrics(y_true_lst, y_pred_lst):
    f, p, r, acc = [], [], [], []
    for y_true, y_pred in zip(y_true_lst, y_pred_lst):
        f.append(f1_score(y_true, y_pred, labels=[0, 1]))
        p.append(precision_score(y_true, y_pred, labels=[0, 1]))
        r.append(recall_score(y_true, y_pred, labels=[0, 1]))
        acc.append(accuracy_score(y_true, y_pred))
    f = np.average(f)
    p = np.average(p)
    r = np.average(r)
    acc = np.average(acc)
    return f, p, r, acc
