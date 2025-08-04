# 🧠 Sentiment Analysis with Word2Vec & RNNs + DCGAN on PathMNIST

This repository implements two advanced deep learning tasks:

1. **Sentiment classification** on the IMDB movie review dataset using Word2Vec and RNN-based architectures (RNN, LSTM, GRU, BiLSTM).
2. **Image generation** on the PathMNIST dataset using a **Deep Convolutional GAN (DCGAN)**.

> 🔍 Focus: Sequence modeling with word embeddings and time series networks + GAN-based synthetic data generation and evaluation via FID score.

---

## 🛠️ Tech Stack

* Python 3.10
* TensorFlow/Keras (for NLP models)
* PyTorch (for GANs)
* Gensim Word2Vec
* MedMNIST (PathMNIST dataset)
* Scikit-learn, Matplotlib, Seaborn

---

## ✅ Part 1: IMDB Sentiment Classification using Word2Vec + RNN Variants

### 📦 Dataset

* Kaggle IMDB dataset: 50,000 labeled reviews
* Preprocessing:

  * HTML stripping, lowercase conversion
  * Tokenization + Stopword removal
  * Lemmatization + contraction expansion

### 🧠 Embedding

* Used **Gensim-trained Word2Vec (CBOW)** model on training tokens
* Created embedding matrix for Keras Embedding layer

### 🧪 Models Trained:

#### 🔹 Simple RNN

* Architecture: Embedding → SimpleRNN(128) → Dense(1, sigmoid)
* EarlyStopping (patience=8)
* ⛔ Weakest performance overall
* 📊 Accuracy: 68.08%, F1: 65.44%

#### 🔹 LSTM

* Architecture: Embedding → LSTM(128, dropout=0.2) → Dense(1, sigmoid)
* 🏆 Strong recall performance: **Recall = 88.96%**, F1 = 87.07%
* Accuracy: 86.79%

#### 🔹 GRU

* Architecture: Embedding → GRU(128, dropout=0.2) → Dense(1, sigmoid)
* 📌 Precision-focused: **Precision = 88.88%**, F1 = 86.33%
* Accuracy: 86.71%

#### 🔹 BiLSTM

* Architecture: Embedding → Bidirectional(LSTM(128, dropout=0.2)) → Dense(1, sigmoid)
* ✅ Best overall: Accuracy = 86.95%, F1 Score = **86.97%**
* Balanced precision and recall

### 📊 Final Results Table

| Model  | Accuracy | Precision | Recall | F1 Score     |
| ------ | -------- | --------- | ------ | ------------ |
| RNN    | 68.08%   | 71.34%    | 60.44% | 65.44%       |
| LSTM   | 86.79%   | 85.26%    | 88.96% | 87.07%       |
| GRU    | 86.71%   | 88.88%    | 83.92% | 86.33%       |
| BiLSTM | 86.95%   | 86.84%    | 87.10% | **86.97%** ✅ |

---

## ✅ Part 2: DCGAN for Medical Image Synthesis (PathMNIST)

### 🩺 Dataset

* **PathMNIST** from MedMNIST: 28x28 RGB microscopy tissue patches
* Split: Train/Test
* Transformed using normalization: \[-1, 1]

### 🎯 DCGAN Setup

* **Generator**:

  * Linear(z\_dim → 1024×4×4) → ConvTranspose2D blocks
  * Hidden Layers: 1024 → 512 → 256 → 128 → 3 (RGB)
  * Activation: ReLU (hidden), Tanh (output)

* **Discriminator**:

  * 5 Conv2D layers + LeakyReLU(0.2) + BatchNorm
  * Output: Fully connected layer with BCEWithLogitsLoss

### 🧪 Training

* Optimizer: Adam (lr = 0.0002, betas=(0.5, 0.999))
* Epochs: 1000 (early stopping at epoch **56**)
* Loss Curves: Generator loss increased as Discriminator stabilized
* No evidence of mode collapse

### 📈 FID Score Evaluation

* Generated 1000 fake + 1000 real images
* Calculated **Fréchet Inception Distance (FID)** using `pytorch-fid`
* 📊 **FID Score: 102.12**

### 🖼️ Visual Inspection

* Generated images (n=32) show texture/color diversity
* Gradual improvement in visual quality over epochs
* Generator converged smoothly

---

## 📈 Summary Tables

### Sentiment Classification (IMDB)

* Word2Vec + RNN Variants
* BiLSTM delivered highest F1 and best balance

### GAN (PathMNIST)

* DCGAN with 5-layer discriminator, 4-layer generator
* FID score used for evaluation (no collapse detected)

---

## 🏷️ Tags

`Sentiment Analysis` `IMDB Dataset` `Word2Vec` `RNN` `LSTM` `GRU` `BiLSTM`
`GAN` `DCGAN` `FID` `PyTorch` `TensorFlow` `Keras` `PathMNIST` `MedMNIST` `Deep Learning Projects`
