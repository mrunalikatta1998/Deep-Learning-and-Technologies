# 🧠 Autoencoders for Fraud Detection & Variational Autoencoder for Fashion MNIST

Implemented and compared multiple Autoencoder Architectures for unsupervised anomaly detection on the Kaggle Credit Card Fraud Detection dataset and explores latent representation learning via a Variational Autoencoder (VAE) on the Fashion MNIST dataset.

> 🔍 Focus: Autoencoder-based anomaly detection, deep and sparse encoding, denoising, and generative representation learning via VAE.

---

## 🛠️ Tech Stack
- Python 3.10
- TensorFlow / Keras
- Scikit-learn
- Matplotlib, Seaborn
- Dataset: Kaggle Credit Card Fraud Detection, Fashion MNIST

---

## ✅ Highlights

### 🔹 1. Vanilla Autoencoder – Credit Card Fraud Detection
- Objective: Train a basic 1-layer autoencoder on normal (non-fraud) transactions to reconstruct input and identify fraud via reconstruction error.
- Architecture:
  - Encoder: Dense(16, ReLU)
  - Decoder: Dense(29, Linear)
- Training:
  - Epochs: 50, Batch size: 128
  - Loss: MSE, Optimizer: Adam
  - EarlyStopping on validation loss
- Evaluation:
  - Threshold set at 95th percentile of reconstruction error
  - ✅ F1 Score: 65.03%, Precision: 82.69%, Recall: 53.55%
- 📊 Reconstruction error plots show clear separation between normal and fraud cases

---

### 🔹 2. Deep Autoencoder – Fraud Detection with Regularization
- Extended architecture with 3-layer encoder and decoder
- Encoder: Dense(128 → 64 → 32), all ReLU + Dropout(0.2–0.4)
- Decoder: Symmetrical Dense layers to reconstruct back to 29 features
- Results:
  - ✅ F1 Score: 65.47%, Precision: 82.71%, Recall: 53.59%
  - Slightly better than Vanilla AE, deeper compression yielded tighter reconstruction
- Visuals:
  - 📈 Epoch-wise training & validation loss
  - 📉 Distribution histograms of reconstruction error for normal vs fraud

---

### 🔹 3. Sparse Autoencoder – L1 Regularization for Anomaly Isolation
- Same architecture as Deep AE, with added `activity_regularizer=regularizers.l1(1e-5)` on the bottleneck layer
- Goal: Enforce sparsity in representations to highlight anomalies
- Results:
  - ✅ Best performing model overall
  - F1 Score: 66.88%, Precision: 85.00%, Recall: 54.94%
  - Correctly detected 437 of 492 frauds (88.82% recall)
- 🧠 Regularization significantly improved minority class detection

---

### 🔹 4. Denoising Autoencoder – Robustness to Input Noise
- Input was corrupted using `GaussianNoise(stddev=0.2)`
- Network trained to reconstruct clean signal from noisy input
- Similar structure to Deep AE
- Results:
  - F1 Score: 65.74%, Precision: 84.74%, Recall: 54.07%
  - Best robustness among all models, especially on borderline anomalies
- Visualizations:
  - 📊 Error histograms: clean vs noisy input
  - 📈 Performance vs threshold graphs

---

### 🔹 5. Model Evaluation & Comparison
- All 4 models used a 95th percentile threshold for evaluation
- Reported:
  - Precision, Recall, F1-score
  - Number of correct fraud detections out of 492
- ✅ Best Overall: **Sparse Autoencoder**
  - Detected 437/492 frauds
  - Balanced performance across metrics
- 📋 Summary Table of architectures, hyperparameters, metrics

---

### 🔹 6. Variational Autoencoder (VAE) – Fashion MNIST
- Objective: Learn 2D latent space encoding of Fashion MNIST images
- Encoder:
  - 4×Conv2D (32→256 filters), each with ReLU + MaxPooling2D
  - Outputs two heads: mean and log variance
- Latent dimension: 2
- Decoder:
  - 1×Dense + Reshape + 4×Conv2DTranspose
  - Final output: 28×28×1 with sigmoid activation
- Training:
  - Custom VAE loss: KL Divergence + Reconstruction (MSE)
  - 100+ epochs with EarlyStopping
- 🧪 Result:
  - Very low reconstruction loss (~5.9e-11)
  - Successful convergence with tight clustering in latent space

---

### 🔹 7. VAE Latent Space Clustering (Visualization)
- Latent embeddings for all 10 classes were projected in 2D space
- Observations:
  - ✅ Clear class-wise clusters despite just 2D space
  - ✅ Most clusters are non-overlapping
  - 🧠 Expected tight packing due to KL regularization
- Visuals:
  - 📉 2D scatter plot of latent embeddings, colored by class label
  - 📊 Mean and Std of latent distributions

---

## 📈 Summary Table – Autoencoders

| Model         | Precision | Recall | F1 Score | Fraud Detected |
|---------------|-----------|--------|----------|----------------|
| Vanilla AE    | 82.69%    | 53.55% | 65.03%   | 263 / 492      |
| Deep AE       | 82.71%    | 53.59% | 65.47%   | 264 / 492      |
| Sparse AE     | 85.00%    | 54.94% | 66.88%   | 437 / 492 ✅    |
| Denoising AE  | 84.74%    | 54.07% | 65.74%   | 266 / 492      |

---

## 🏷️ Tags
`Autoencoder` `Anomaly Detection` `Fraud Detection` `Sparse AE` `Denoising AE` `VAE`  
`Credit Card Dataset` `Fashion MNIST` `Latent Space Clustering` `Reconstruction Loss` `TensorFlow`
