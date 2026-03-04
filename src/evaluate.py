import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
)

from config import MODELS_DIR, THRESHOLD
from dataset import get_datasets


def evaluate():
    model_path = str(MODELS_DIR / "best_model.keras")
    model = tf.keras.models.load_model(
    model_path,
    safe_mode=False
)
    print(f"Loaded model from {model_path}\n")


    _, val_ds = get_datasets()


    y_true, y_pred_prob = [], []
    for X_batch, y_batch in val_ds:
        probs = model.predict(X_batch, verbose=0).flatten()
        y_pred_prob.extend(probs.tolist())
        y_true.extend(y_batch.numpy().tolist())

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    y_pred = (y_pred_prob >= THRESHOLD).astype(int)

   
    print("=" * 50)
    print(f"Threshold: {THRESHOLD}")
    print("=" * 50)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["No Mask", "Mask"]))

    roc_auc = roc_auc_score(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")

    print("\nThreshold sweep (0.3 → 0.7):")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for t in np.arange(0.3, 0.75, 0.05):
        preds = (y_pred_prob >= t).astype(int)
        report = classification_report(y_true, preds, output_dict=True,
                                       zero_division=0)
        p = report["1"]["precision"]
        r = report["1"]["recall"]
        f1 = report["1"]["f1-score"]
        print(f"{t:>10.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")


if __name__ == "__main__":
    evaluate()