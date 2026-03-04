import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from config import IMAGE_SIZE, CHANNELS, LEARNING_RATE


def build_augmentation():
    """Data augmentation applied only during training."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def build_model():
    """
    MobileNetV2 transfer learning model.
    Base frozen → fine-tune top layers only.
    """
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  

    augmentation = build_augmentation()

    inputs = layers.Input(shape=(*IMAGE_SIZE, CHANNELS))
    x = augmentation(inputs)                    
    x = tf.keras.layers.Lambda(preprocess)(x)   
    x = base(x, training=False)                   
    x = layers.GlobalAveragePooling2D()(x)        
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
