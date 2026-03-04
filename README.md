# Face Mask Detection using CNN

This project implements a binary image classification model to detect
whether a person is wearing a face mask or not using a Convolutional
Neural Network (CNN).

---

## Project Structure
FACE_MASK_DETECTION/
│
├── data/
├── models/
├── src/
│   ├── cnn_simple.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   └── training.py
├── venv/
├── .gitignore
├── README.md
└── requirements.txt

---

## Dataset
Dataset from kaggle

https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

The dataset consists of two classes:
- `with_mask`
- `without_mask`

Images are resized and normalized before being passed to the model.

---

## Model

A CNN-based architecture was used for binary classification.
The model outputs probabilities, allowing flexible threshold tuning
during evaluation.

---

## Training

To train the model:

```bash
python src/training.py
```

## Evaluation

The trained model is saved in the `models/` directory.

### Model Performance Metrics

Model performance is evaluated using metrics beyond accuracy, including:

- Confusion Matrix  
- Precision, Recall, F1-score  
- ROC-AUC  
- PR-AUC  
- Classification Threshold Tuning  

### Threshold Tuning

Different thresholds (0.3–0.7) were evaluated.  
Based on the balance between precision and recall, **0.6** was selected as the final operating threshold.

### Final Results (Threshold = 0.6)

- Accuracy: **91%**  
- Macro F1-score: **0.91**  
- ROC-AUC: **0.97**  
- PR-AUC: **0.97**

These results indicate strong discriminative ability and robust performance across different thresholds.

## Usage

Run evaluation:
python src/evaluate.py


Notes

Accuracy alone was not considered sufficient. Threshold tuning and
threshold-independent metrics were used to better reflect real-world
performance.
