from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def build_model(input_shape):
    """ Inputs """
    inputs = L.Input(input_shape)

    """ Backbone """
    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        alpha=1.0
    )
    backbone.trainable = True
    # backbone.summary()

    """ Detection Head """
    x = backbone.output
    x = L.Conv2D(256, kernel_size=1, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(4, activation="sigmoid")(x)

    """ Model """
    model = Model(inputs, x)
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_model(input_shape)
    model.summary()
