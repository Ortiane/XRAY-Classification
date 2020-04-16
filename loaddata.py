#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import csv

def makeDataset(x_path, csv_filename, batch_size):
    x_file_list = []
    y_str_list = []
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                line_count += 1
                x_file_list.append(x_path + row[0])
                y_str_list.append(row[1].split('|'))

    y_dict = {
        'Emphysema': 0, 
        'Infiltration': 1, 
        'Pleural_Thickening': 2, 
        'Pneumothorax': 3, 
        'Cardiomegaly': 4, 
        'Atelectasis': 5, 
        'Edema': 6, 
        'Effusion': 7, 
        'Consolidation': 8, 
        'Mass': 9, 
        'Nodule': 10, 
        'Fibrosis': 11, 
        'Pneumonia': 12, 
        'Hernia': 13
    }

    y_list = []
    for i in range(len(y_str_list)):
        labels = [0] * 14
        for y_logit in y_str_list[i]:
            if y_logit == 'No Finding':
                continue
            labels[y_dict[y_logit]] = 1
        labels = tf.constant(labels, dtype=tf.float32)
        y_list.append(labels)
    dataset = tf.data.Dataset.from_tensor_slices((x_file_list, y_list))

    def parse_fn(filename, label):
        img_str = tf.io.read_file(filename)
        img_dcd = tf.image.decode_png(img_str, channels=3)
        img_dcd = tf.image.resize(img_dcd, [224, 224])
        return img_dcd, label

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

if __name__ == '__main__':
    x_path = '../sample/images/'
    csv_file = '../sample/sample_labels.csv'
    batch_size = 3
    dataset = makeDataset(x_path, csv_file, batch_size)
    for x, y in dataset:
        print(x.shape, y.shape)
        break
