# 🧠 Deep Learning Exploration: Computer Vision, Anomaly Detection, NLP & GANs

This repository presents a curated collection of deep learning projects across multiple real-world domains including image classification, anomaly detection, natural language processing, and generative modeling. Each module includes end-to-end pipelines involving data preprocessing, model design, training, evaluation, visualization, and performance tuning.

> 🧪 Focused on practical applications of deep learning using TensorFlow, Keras, PyTorch, and Word2Vec — with attention to explainability, generalization, and interpretability.

---

### 🔢 1. MNIST Digit Classification with ANN Architectures
- Designed multiple feedforward neural networks for classifying digits using the MNIST dataset.
- Compared effects of different weight initializers (He, Xavier, Normal) and activation functions (ReLU, Tanh, Sigmoid).
- Evaluated binary classification (digits 5 vs 8) and full multi-class setup (digits 0–9).
- Techniques: Manual backpropagation, confusion matrices, overfitting mitigation, hyperparameter tuning.

---

### 🩺 2. Medical Imaging with CNNs, Transfer Learning, Grad-CAM & Quantization
- Applied 3D CNN on FractureMNIST3D for rib fracture classification (volumetric CT scans).
- Built a deep CNN and used transfer learning (VGG-19) for 7-way skin lesion classification using DermaMNIST.
- Enhanced model interpretability with Grad-CAM heatmaps for saliency visualization.
- Compressed models using quantization techniques (dynamic, integer, QAT) for deployment readiness.
- Techniques: 3D/2D ConvNets, data augmentation, class weighting, F1 analysis, explainability.

---

### 💳 3. Anomaly Detection in Financial Transactions with Autoencoders & VAE
- Built and compared four Autoencoder variants for fraud detection:
  - Vanilla AE, Deep AE, Sparse AE (L1 regularization), and Denoising AE.
- Trained models only on normal transactions; evaluated using reconstruction error on full test set.
- Calculated precision, recall, and F1 scores to measure fraud detection effectiveness.
- Applied Variational Autoencoder (VAE) to Fashion MNIST for learning compressed latent space representations.
- Visualized 2D latent space clusters and generated reconstructed images using Conv2D and Conv2DTranspose.
- Techniques: Unsupervised anomaly detection, reconstruction thresholding, KL-divergence, latent space learning.

---

### 📝 4. Sentiment Classification + Generative Adversarial Networks (DCGAN)
- Implemented sentiment analysis on IMDB movie reviews using Gensim Word2Vec + RNN variants:
  - RNN, LSTM, GRU, and BiLSTM — with comparative evaluation of precision, recall, F1.
- Built a DCGAN to generate synthetic tissue images using PathMNIST dataset.
  - Generator and Discriminator trained with adversarial loss for 1000+ epochs.
  - Evaluated using FID (Fréchet Inception Distance) and visual inspection for mode collapse.
- Techniques: NLP preprocessing, sequence modeling, GAN training dynamics, FID evaluation.

---

## 📈 Summary of Techniques Applied

- ✅ Neural Network architecture design & backpropagation
- ✅ Convolutional Neural Networks (2D & 3D)
- ✅ Transfer learning (VGG-19), data augmentation, Grad-CAM
- ✅ Autoencoders, sparsity constraints, denoising, and VAE
- ✅ Word2Vec embedding + RNN-based sentiment classifiers
- ✅ DCGANs and synthetic data generation with FID scoring
- ✅ Quantization (for efficient deployment)

---

## 🛠️ Tools & Libraries

- **TensorFlow / Keras** – supervised models, autoencoders, VAE
- **PyTorch** – for DCGAN and custom training loops
- **Gensim** – Word2Vec for NLP embeddings
- **Scikit-learn** – metrics, evaluations, preprocessing
- **MedMNIST** – medical image datasets (PathMNIST, DermaMNIST, FractureMNIST3D)
- **Matplotlib / Seaborn** – visualization

---

## 🏷️ Tags

`Deep Learning` `Computer Vision` `Autoencoders` `GANs` `Sentiment Analysis` `VAE`  
`Word2Vec` `PyTorch` `TensorFlow` `Explainable AI` `Quantization` `MedMNIST` `IMDB Reviews`
