# Covid-19 Chest X-Ray Classifier — CNN Image Classification
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-grey?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-blue)

A convolutional neural network built with TensorFlow/Keras that classifies chest X-ray images into three diagnostic categories — **Covid-19, Viral Pneumonia, and Normal** — achieving over 86% accuracy on unseen test data.

---

## Project Overview

This project trains a deep learning CNN on chest X-ray images to assist in the diagnostic classification of lung conditions. The model is trained on the Covid-19 Image Dataset sourced from Kaggle and applies real-world medical imaging techniques including grayscale preprocessing, data augmentation, and early stopping with best weight restoration. Model performance is evaluated using a full classification report and a confusion matrix heatmap.

---

## Dataset

**Source:** [Covid-19 Image Dataset — Kaggle (pranavraikokte)](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

| Class | Description |
|---|---|
| Covid | Chest X-rays of Covid-19 positive patients |
| Viral Pneumonia | Chest X-rays of Viral Pneumonia patients |
| Normal | Chest X-rays of healthy patients |

The dataset is pre-split into `train/` and `test/` directories, each containing the three class subfolders.

```
Covid19-dataset/
    train/
        Covid/
        Normal/
        Viral Pneumonia/
    test/
        Covid/
        Normal/
        Viral Pneumonia/
```

---

## Model Architecture

```
Input Layer         →  (128, 128, 1) — grayscale X-ray images
Conv2D(32, 3x3)     →  ReLU + same padding
MaxPooling2D        →  (2, 2)
Conv2D(64, 3x3)     →  ReLU + same padding
MaxPooling2D        →  (2, 2)
Conv2D(128, 3x3)    →  ReLU + same padding
MaxPooling2D        →  (2, 2)
Flatten
Dropout(0.3)        →  Regularization
Dense(64)           →  ReLU
Dense(3)            →  Softmax output — 3 classes
```

**Loss function:** Sparse Categorical Crossentropy  
**Optimizer:** Adam (learning_rate=0.0005)  
**Metric:** Sparse Categorical Accuracy

---

## Features

- **Grayscale preprocessing** — X-rays loaded as single channel images `(128, 128, 1)`
- **Data augmentation** — rotation, shifts, zoom and horizontal flip applied to training data only
- **Separate generators** — augmentation applied to training data, rescaling only to test data
- **Early stopping** — monitors `val_loss` with `patience=5` and `restore_best_weights=True`
- **Dropout regularization** — reduces overfitting on a relatively small medical dataset
- **Full evaluation pipeline** — classification report, confusion matrix and training curve visualisations

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 86%+ |
| Loss Function | Sparse Categorical Crossentropy |
| Early Stopping | Triggered based on val_loss |

---

## Training Diagnostics

The model generates accuracy and loss curves across all training epochs for both training and validation data.

![Training Curves](static/images/my_plots.png)

---

## Confusion Matrix

![Confusion Matrix](static/images/confusion_matrix.png)

---

## Why Precision and Recall Matter Here

For a medical classification model, accuracy alone is insufficient. In Covid-19 detection:

- **Precision** — of all predicted Covid cases, how many were actually Covid
- **Recall** — of all actual Covid cases, how many did the model correctly identify

A high Recall is clinically critical — missing a Covid case (false negative) carries far greater risk than a false alarm (false positive).

---

## Getting Started

### Prerequisites

```bash
pip install tensorflow scikit-learn matplotlib seaborn
```

### Run the classifier

```bash
python script.py
```

### Expected output
- Training progress printed per epoch
- Classification report printed to console
- `static/images/my_plots.png` — training curves
- `static/images/confusion_matrix.png` — confusion matrix heatmap

---

## Concepts Demonstrated

- Convolutional Neural Network architecture for image classification
- Medical image preprocessing with grayscale normalisation
- Data augmentation for small medical datasets
- Separate train/test data generators to prevent data leakage
- Early stopping with best weight restoration
- Model evaluation with classification report and confusion matrix
- Training diagnostics visualisation with Matplotlib

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
