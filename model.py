from tensorflow import keras
import tensorflow as tf


class Modified_ResNet50(keras.models.Model):
    def __init__(self, output_size=14):
        super(Modified_ResNet50, self).__init__()
        self.output_size = output_size

    def build(self, input_shape=(224, 224, 3)):
        self.resnet = keras.applications.resnet_v2.ResNet50V2(
            include_top=False, weights=None, input_shape=input_shape
        )
        self.pooling_layer = keras.layers.GlobalAveragePooling2D()
        self.output_layer = keras.layers.Dense(self.output_size)
        self.model = tf.keras.Sequential(
            [self.resnet, self.pooling_layer, self.output_layer]
        )
        super(Modified_ResNet50, self).build(input_shape)

    def call(self, inputs):
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, 0)
        return self.model(inputs)

if __name__ == '__main__':
    model = Modified_ResNet50()
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    model.build()
    model.summary()
