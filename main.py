import tensorflow as tf
from loaddata import makeDataset
from model import Modified_ResNet50

def train(net, dataset, epochs):
    for epoch_idx in range(epochs):
        print(epoch_idx)
        net.fit(dataset,
            verbose=2)
    return net

def main():
    net = Modified_ResNet50()
    net.compile(
        loss=tf.nn.sigmoid_cross_entropy_with_logits,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.nn.sigmoid_cross_entropy_with_logits],
    )
    net.build()
    net.summary()
    x_path = '../sample/images/'
    csv_file = '../sample/sample_labels.csv'
    batch_size = 3
    dataset = makeDataset(x_path, csv_file, batch_size)
    epochs = 1

    trained_net = train(net, dataset, epochs)

if __name__ == '__main__':
    main()