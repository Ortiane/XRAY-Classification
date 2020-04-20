from tensorflow import keras
import efficientnet.tfkeras as efn
import tensorflow as tf

# Install efficientnet using "pip install -U --pre efficientnet"
# Add the mobilenet_v3 file into the keras-applications in .conda\envs\11695\Lib\site-packages


class Model(keras.models.Model):
    def __init__(self, output_size=15, model_type="efficientnet"):
        super(Model, self).__init__()
        self.output_size = output_size
        self.model_type = model_type

    def build(self, input_shape=(224, 224, 3)):
        if self.model_type == "resnet":
            self.default_model = keras.applications.resnet_v2.ResNet50V2(
                include_top=False, weights=None, input_shape=input_shape
            )
        elif self.model_type == "efficientnet":
            self.default_model = efn.EfficientNetB4(
                include_top=False, weights=None, input_shape=input_shape
            )
        elif self.model_type == "mobilenet":
            self.default_model = keras.applications.mobilenet_v2.MobileNetV2(
                alpha=1.4, include_top=False, weights=None, input_shape=input_shape
            )
        self.pooling_layer = keras.layers.GlobalAveragePooling2D()
        self.dropout_layer = keras.layers.Dropout(0.2)
        self.output_layer = keras.layers.Dense(self.output_size)
        self.activation = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        self.model = tf.keras.Sequential(
            [
                self.default_model,
                self.pooling_layer,
                self.dropout_layer,
                self.output_layer,
                self.activation,
            ]
        )
        super(Model, self).build(input_shape)

    def call(self, inputs):
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, 0)
        return self.model(inputs)

if __name__ == "__main__":
    model = Model()
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    model.build()
    model.summary()
