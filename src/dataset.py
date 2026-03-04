import tensorflow as tf
from config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, RANDOM_STATE


def get_datasets():
    """
    Returns (train_ds, val_ds, test_ds) as tf.data pipelines.
    Images are streamed from disk — no full RAM load.
    """
    # Train + validation split (80/20)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=RANDOM_STATE,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_STATE,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds