import argparse
import datetime
import os

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
    parser.add_argument("--preprocessed_split", default=4, type=int)
    parser.add_argument("--log_dir", default="runs", type=str)
    parser.add_argument("--train_txt",default="data/train_val_list.txt", type=str)
    parser.add_argument("--test_txt",default="data/test_list.txt", type=str)
    args = parser.parse_args()
    return args


def train(net, dataset, testdataset, epochs, model_save_dir, logdir):
    log_dir = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    mc = tf.keras.callbacks.ModelCheckpoint(
        model_save_dir + "model.h5",
        monitor="loss",
        mode="max",
        verbose=1,
        save_best_only=True,
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

    # dataset = makeDatasetPreprocessed(
    #     X_PATH, CSV_FILE, BATCH_SIZE, PREPROCESSED, PREPROCESSED_SPLIT
    # )
    train_dataset = makeDatasetPreprocessed(
        X_PATH, CSV_FILE, BATCH_SIZE, PREPROCESSED, PREPROCESSED_SPLIT,args.train_txt
    )
    print(train_dataset)
    test_dataset = makeDatasetPreprocessed(
        X_PATH, CSV_FILE, BATCH_SIZE, PREPROCESSED, PREPROCESSED_SPLIT,args.test_txt
    )
    print(test_dataset)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        net = Model(model_type="mobilenet")
        net.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["binary_accuracy"],
        )
        net.build()
        net.summary()

        trained_net = train(net, train_dataset, test_dataset, EPOCHS, MODEL_DIR, LOG_DIR)


if __name__ == "__main__":
    main()
