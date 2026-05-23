# Covid-19 Chest X-Ray Classifier — CNN Image Classification
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.14.0-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.0-11557c?style=flat&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13+-4C72B0?style=flat&logoColor=white)](https://seaborn.pydata.org/)

A convolutional neural network built with TensorFlow/Keras that classifies chest X-ray images into four diagnostic categories — **Covid-19, Viral Pneumonia, Lung Opacity and Normal** — trained on over 15,000 chest X-ray images from the COVID-19 Radiography Database.

---

## Project Overview

This project trains a deep learning CNN on chest X-ray images to assist in the diagnostic classification of lung conditions. The model is trained on the COVID-19 Radiography Database sourced from Kaggle — a medically supervised dataset of over 15,000 chest X-rays across four classes. The pipeline applies real-world medical imaging techniques including grayscale preprocessing, aggressive data augmentation, class weight balancing for imbalanced classes, and early stopping with best weight restoration. Model performance is evaluated using a full classification report and a confusion matrix heatmap.

---

## Dataset

**Source:** [COVID-19 Radiography Database — Kaggle (tawsifurrahman)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

| Class | Images | Description |
|---|---|---|
| Normal | 10,192 | Chest X-rays of healthy patients |
| COVID | 3,616 | Chest X-rays of Covid-19 positive patients |
| Lung Opacity | 6,012 | Chest X-rays showing non-COVID lung opacity |
| Viral Pneumonia | 1,345 | Chest X-rays of Viral Pneumonia patients |

The dataset is pre-split into `train/` and `test/` directories, each containing the three class subfolders.

```
Covid19_Radiography_Dataset/
    COVID/
        images/
        mask/
    Normal/
        images/
        mask/
    Lung_Opacity/
        images/
        mask/
    Viral Pneumonia/
        images/
        mask/
```

---

## Model Architecture

```
Input Layer         →  (224, 224, 1) — grayscale X-ray images
Conv2D(32, 3x3)     →  ReLU + same padding
Batch Normalization
MaxPooling2D        →  (2, 2)
Conv2D(64, 3x3)     →  ReLU + same padding
BatchNormalization
MaxPooling2D        →  (2, 2)
Conv2D(128, 3x3)    →  ReLU + same padding
BatchNormalization
MaxPooling2D        →  (2, 2)
Flatten
Dropout(0.5)        →  Regularization
Dense(64)           →  ReLU
Dropout(0.3)        →  Regularization
Dense(4)            →  Softmax output — 4 classes
```

**Loss function:** Sparse Categorical Crossentropy  
**Optimizer:** Adam (learning_rate=0.0005)  
**Metric:** Sparse Categorical Accuracy

---

## Features

- **Grayscale preprocessing** — X-rays loaded as single channel images `(224, 224, 1)`
- **Custom train/test split** — dataset split 80/20 with stratification across all 4 classes
- **Data augmentation** — rotation, shifts, zoom and horizontal flip and brightness variation applied to training data only
- **Separate generators** — augmentation applied to training data, rescaling only to test data prevents data leakage
- **Class weight balancing** — computed class weights address severe imbalance between Normal (10,192) and Viral Pneumonia (1,345)
- **Batch Normalization** — stabilises training across all three convolutional blocks
- **Early stopping** — monitors `val_loss` with `patience=5` and `restore_best_weights=True`
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

A high Recall is clinically critical — missing a Covid case (false negative) carries far greater risk than a false alarm (false positive). This is reflected in the class weight balancing strategy, which penalises the model more heavily for misclassifying minority classes.

---

## Getting Started

### Prerequisites

```bash
pip install tensorflow scikit-learn matplotlib seaborn
```

### Dataset Setup
1. Download the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle
2. Extract to your project directory as `COVID-19_Radiography_Dataset/`
3. Uncomment the `create_train_test_split()` call in `script.py` and run once to create the train/test split
4. Comment it back out before rerunning

### Run the classifier

```bash
python script.py
```

### Expected output
- Train/test split created in `Covid19-dataset/`
- Training progress printed per epoch
- Classification report printed to console
- `static/images/my_plots.png` — training curves
- `static/images/confusion_matrix.png` — confusion matrix heatmap

---

## Concepts Demonstrated

- Convolutional Neural Network architecture for image classification
- Medical image preprocessing with grayscale normalisation
- Custom dataset splitting for datasets without pre-built train/test structure
- Data augmentation to improve generalisation on medical imaging data
- Class weight balancing for imbalanced multi-class datasets
- Early stopping with best weight restoration
- Model evaluation with classification report and confusion matrix
- Training diagnostics visualisation with Matplotlib

---

## License

This project is open source and available under the [MIT License](LICENSE).
