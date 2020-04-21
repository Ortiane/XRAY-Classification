import argparse
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, confusion_matrix
import itertools

import h5py
import tensorflow as tf
# import tensorflow.data.Dataset as tfds

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
    parser.add_argument("--train_txt",default="data/train_val_list.txt", type=str)
    parser.add_argument("--test_txt",default="data/test_list.txt", type=str)
    parser.add_argument('--checkpoint_path', default="save/model.h5")
    parser.add_argument('--model_name', default="mobilenet")
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

labels = [
        "Emphysema",
        "Infiltration",
        "Pleural_Thickening",
        "Pneumothorax",
        "Cardiomegaly",
        "Atelectasis",
        "Edema",
        "Effusion",
        "Consolidation",
        "Mass",
        "Nodule",
        "Fibrosis",
        "Pneumonia",
        "Hernia",
        # "No Finding"
    ]

def test(net, dataset, model_name):
    # log_dir = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # mc = tf.keras.callbacks.ModelCheckpoint(
    #     model_save_dir + "model.h5",
    #     monitor="loss",
    #     mode="max",
    #     verbose=1,
    #     save_best_only=True,
    # )
    # tc = tf.keras.callbacks.TensorBoard(
    #     log_dir=log_dir, update_freq=25, write_graph=False, profile_batch=0
    # )

    # net.fit(
    #     dataset,
    #     epochs=epochs,
    #     validation_data=testdataset,
    #     callbacks=[mc, tc],
    #     shuffle=True,
    # )

    # # return net
    # for x, y in dataset:
    #     y_pred = net.predict(x)
    #     print(y_pred)
    #     input()
    #     print(y)
    #     input()
    #     print(multiplication_f_loss(y, y_pred))
    #     print(recall(y, y_pred))
    #     input()
    # return y_pred

    # res = net.evaluate(dataset)


    # # print ROC
    # test_X, test_Y = next(create_data_generator(
    #     valid, labels, 10000, None, target_size=input_shape))
    # pred_Y = model.predict(test_X, batch_size=32, verbose=True)
    
    y_pred = net.predict(dataset)
    y_true = []
    i = 0
    for x, y in dataset:
        y_true.append(y)
        # y_pred.append(net.predict(x))
        # print(net.predict(x))
        # input()
        # print(i)
        i += 1
    y_true = np.concatenate(y_true)

    plot_ROC(labels, y_true, y_pred, model_name)


#   # Calculate the confusion matrix.
    # print(np.argmax(y_true,axis=1))
    # print(np.argmax(y_pred,axis=1))
    cm = confusion_matrix(np.argmax(y_true,axis=1),np.argmax(y_pred,axis=1))
   # cm = multilabel_confusion_matrix(y_true, tf.round(y_pred))
    plot_confusion_matrix(cm, labels, model_name)



    # return res

def plot_ROC(labels, test_Y, pred_Y, model_name='mobilenet'):
    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    for (idx, c_label) in enumerate(labels):
        fpr, tpr, thresholds = roc_curve(
            test_Y[:, idx].astype(int), pred_Y[:, idx])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_title(model_name+' ROC Curve')
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')

    ROC_image_file_path = os.path.join(
        model_name + '_ROC.png')

    fig.savefig(ROC_image_file_path)
    print('Saved ROC plot at'+ROC_image_file_path)



def plot_confusion_matrix(cm, class_names, model_name='mobilenet'):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  np.seterr(divide='ignore', invalid='ignore')
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  
  cm_image_file_path = os.path.join(
        model_name + '_CM.png')

  figure.savefig(cm_image_file_path)
  print('Saved ROC plot at'+cm_image_file_path)

# def log_confusion_matrix(epoch, logs):
#   # Use the model to predict the values from the validation dataset.
#   test_pred_raw = model.predict(test_images)
#   test_pred = np.argmax(test_pred_raw, axis=1)

#   # Calculate the confusion matrix.
#   cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
#   # Log the confusion matrix as an image summary.
#   figure = plot_confusion_matrix(cm, class_names=labels)
#   cm_image = plot_to_image(figure)

#   # Log the confusion matrix as an image summary.
#   with file_writer_cm.as_default():
#     tf.summary.image("Confusion Matrix", cm_image, step=epoch)




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
    CHECKPOINT_PATH = args.checkpoint_path
    MODEL_NAME = args.model_name


    test_dataset = makeDatasetPreprocessed(
        X_PATH, CSV_FILE, BATCH_SIZE, PREPROCESSED, PREPROCESSED_SPLIT,args.test_txt
    )
    print(test_dataset)

    strategy = tf.distribute.MirroredStrategy()
    # logdir = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
    # # Define the per-epoch callback.
    # cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    with strategy.scope():
        net = Model(model_type=MODEL_NAME)
        net.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["binary_accuracy"],
        )
        net.build()
        net.summary()
        net.load_weights(CHECKPOINT_PATH)

        y_pred = test(net, test_dataset, MODEL_NAME)

    #     print(y_pred[0])

    # import pdb
    # pdb.set_trace()
        # y_pred_np = tfds.as_numpy(y_pred)
        # xy = tfds.as_numpy()


if __name__ == "__main__":
    main()
