import argparse

import h5py
import tensorflow as tf

from loaddata import *
from model import Model


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
    parser.add_argument("--log_dir", default="runs", type=str)
    args = parser.parse_args()
    return args


def train(net, dataset, epochs, model_save_dir, logdir):
    mc = tf.keras.callbacks.ModelCheckpoint(
        model_save_dir + 'model.h5', monitor="loss", mode="max", verbose=1, save_best_only=True,
    )
    tc = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    net.fit(dataset, epochs=epochs, validation_data=dataset, callbacks=[mc, tc])

    return net


def main():
    args = parse_args()
    X_PATH = args.image_dir
    CSV_FILE = args.csv_file
    MODEL_DIR = args.model_save_dir
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LOG_DIR = args.log_dir
    PREPROCESSED = str2bool(args.preprocessed)

    dataset = makeDatasetPreprocessed(X_PATH, CSV_FILE, BATCH_SIZE, PREPROCESSED)

    net = Model()
    net.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["binary_accuracy"],
    )
    net.build()
    net.summary()

    trained_net = train(net, dataset, EPOCHS, MODEL_DIR, LOG_DIR)


if __name__ == "__main__":
    main()
