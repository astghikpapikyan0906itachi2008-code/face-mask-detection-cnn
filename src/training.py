import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import MODELS_DIR, EPOCHS
from dataset import get_datasets
from cnn_simple import build_model


def train():
    
    os.makedirs(MODELS_DIR, exist_ok=True)


    train_ds, val_ds = get_datasets()


    model = build_model()
    model.summary()


    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

 
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(str(MODELS_DIR / "final_model.keras"))
    print(f"\nModels saved to {MODELS_DIR}")

    return history


if __name__ == "__main__":
    train()