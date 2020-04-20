import argparse
import datetime
import os

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from loaddata import *
from model import *

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="data/preprocessing/images/", type=str)
    parser.add_argument("--csv_file", default="data/sample/sample_labels.csv", type=str)
    parser.add_argument("--model_save_dir", default="save/", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--preprocessed", default="True", type=str)
    parser.add_argument("--preprocessed_split", default=4, type=int)
    parser.add_argument("--log_dir", default="runs", type=str)
    parser.add_argument("--train_txt", default="data/train_val_list.txt", type=str)
    parser.add_argument("--test_txt", default="data/test_list.txt", type=str)
    parser.add_argument("--model_name", default="mobilenet", type=str)
    args = parser.parse_args()
    return args


def train(net, dataset, testdataset, epochs, model_save_dir, logdir, model_name):
    log_dir = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    mc = tf.keras.callbacks.ModelCheckpoint(
        model_save_dir + model_name + "_model.h5",
        monitor="val_binary_accuracy",
        mode="max",
        verbose=1,
        save_best_only=False,
    )
    tc = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq=25, write_graph=False, profile_batch=0
    )

    net.fit(
        dataset,
        epochs=epochs,
        validation_data=testdataset,
        callbacks=[mc, tc],
        shuffle=True,
    )

    return net


# https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py#L90-L97
# unable to use tfa because tensorflow is not 2.0
def sigmoid_focal_crossentropy(
    y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False
):
    """
    Args
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)


# https://github.com/keras-team/keras/issues/6507
def true_pos(y_true, y_pred):
    return K.sum(y_true * K.round(y_pred))


def false_pos(y_true, y_pred):
    return K.sum(y_true * (1.0 - K.round(y_pred)))


def false_neg(y_true, y_pred):
    return K.sum((1.0 - y_true) * K.round(y_pred))


def precision(y_true, y_pred):
    return true_pos(y_true, y_pred) / (
        true_pos(y_true, y_pred) + false_pos(y_true, y_pred) + K.epsilon()
    )


def recall(y_true, y_pred):
    return true_pos(y_true, y_pred) / (
        true_pos(y_true, y_pred) + false_neg(y_true, y_pred) + K.epsilon()
    )


def f1_score(y_true, y_pred):
    return 2.0 / (
        1.0 / recall(y_true, y_pred) + 1.0 / precision(y_true, y_pred) + K.epsilon()
    )


def main():
    args = parse_args()
    X_PATH = args.image_dir
    CSV_FILE = args.csv_file
    MODEL_DIR = args.model_save_dir
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LOG_DIR = args.log_dir
    PREPROCESSED = str2bool(args.preprocessed)
    PREPROCESSED_SPLIT = args.preprocessed_split
    MODEL_NAME = args.model_name

    train_dataset = makeDatasetPreprocessed(
        X_PATH, CSV_FILE, BATCH_SIZE, PREPROCESSED, PREPROCESSED_SPLIT, args.train_txt
    )
    print(train_dataset)
    test_dataset = makeDatasetPreprocessed(
        X_PATH, CSV_FILE, BATCH_SIZE, PREPROCESSED, PREPROCESSED_SPLIT, args.test_txt
    )
    print(test_dataset)
    # dataset = makeDatasetPreprocessed(
    #     X_PATH, CSV_FILE, BATCH_SIZE, PREPROCESSED, PREPROCESSED_SPLIT
    # )
    # test_dataset = train_dataset = dataset
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        net = Model(model_type=MODEL_NAME)
        net.compile(
            loss=sigmoid_focal_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[recall, precision, f1_score],
            # metrics=[recall],
        )
        net.build()
        net.summary()

        trained_net = train(
            net, train_dataset, test_dataset, EPOCHS, MODEL_DIR, LOG_DIR, MODEL_NAME
        )


if __name__ == "__main__":
    main()
