# 🧠 Brain Tumor Classification using EfficientNet-B0 and PyTorch

## Overview

This project is an end-to-end deep learning application for multiclass brain tumour classification using MRI images. The model is built using **EfficientNet-B0** with transfer learning in **PyTorch** and classifies MRI scans into four categories:

* Glioma
* Meningioma
* Pituitary Tumour
* No Tumour

The project follows a modular machine learning engineering workflow, including data preprocessing, model training, evaluation, inference, and deployment through a Streamlit web application.

---

## Features

* EfficientNet-B0 Transfer Learning
* Custom PyTorch Dataset and DataLoader
* Modular project architecture
* Image preprocessing pipeline
* Training and validation engine
* Model checkpoint saving
* Model evaluation with classification report and confusion matrix
* Single-image prediction script
* Interactive Streamlit web application
* Clean and reusable project structure

---

## Project Structure

```text
brain-tumor-classification/

├── app/
│   └── app.py
│
├── checkpoints/
│   └── best_model.pth
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── outputs/
│   ├── figures/
│   └── predictions/
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── engine.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

Dataset: **Brain Tumor MRI Dataset**

Classes:

* Glioma
* Meningioma
* No Tumor
* Pituitary

Dataset Distribution

| Split    | Images |
| -------- | -----: |
| Training |   5600 |
| Testing  |   1600 |
| Total    |   7200 |

The dataset is not included in this repository because of GitHub storage limitations. Download it from Kaggle and place it under:

```text
data/raw/
```

---

## Model

Architecture:

* EfficientNet-B0
* Transfer Learning
* ImageNet Pretrained Weights

Loss Function:

* CrossEntropyLoss

Optimizer:

* Adam

Framework:

* PyTorch

---

## Results

### Training Performance

| Metric              |  Value |
| ------------------- | -----: |
| Training Accuracy   | 99.77% |
| Validation Accuracy | 95.25% |

Classification Report

| Class      | Precision | Recall | F1-score |
| ---------- | --------: | -----: | -------: |
| Glioma     |      0.99 |   0.84 |     0.91 |
| Meningioma |      0.91 |   0.97 |     0.94 |
| No Tumor   |      0.93 |   1.00 |     0.96 |
| Pituitary  |      0.99 |   0.99 |     0.99 |

Overall Test Accuracy:

**95.25%**

---

## Installation

Clone the repository

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/brain-tumor-classification.git

cd brain-tumor-classification
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python -m src.train
```

---

## Model Evaluation

```bash
python -m src.evaluate
```

---

## Single Image Prediction

```bash
python -m src.predict path/to/image.jpg
```

Example Output

```text
Prediction : Pituitary

Confidence : 99.90%
```

---

## Run the Streamlit App

```bash
streamlit run app/app.py
```

Upload an MRI image and the model will display:

* Predicted Class
* Confidence Score
* Probability for each class

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* Streamlit
* NumPy
* Pillow
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

## Future Improvements

* Grad-CAM visualisation
* External dataset validation
* Model quantisation for edge deployment
* Docker containerisation
* CI/CD pipeline using GitHub Actions
* ONNX model export

---

## Disclaimer

This project is intended for educational and research purposes only.

It is **not** a medical diagnostic system and should not be used for clinical decision-making.

---

## Author

**Karthik K Jash**

B.Tech Electronics and Computer Engineering

Amrita Vishwa Vidyapeetham

Interested in Machine Learning, Computer Vision, Embedded AI and Intelligent Systems.
