# 🧠 Exploring Neural Networks: Architecture, Activation & Optimization Techniques using MNIST Digit Dataset

This repository presents a hands-on case study of training and evaluating feedforward neural networks on the MNIST handwritten digit dataset. The goal was to systematically explore the impact of different design and optimization choices—like activation functions, weight initialization strategies, and hyperparameters—on training performance and generalization.

The work combines theoretical insights (manual backpropagation and gradient flow analysis) with empirical experiments, using TensorFlow/Keras to validate and compare multiple network configurations.

> 🔍 Focus: Bridging theory and implementation in neural network design, performance tuning, and interpretability.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **Libraries:** TensorFlow / Keras, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Tools:** Jupyter Notebook

---

## ✅ Highlights

---

### 🔹 1. Manual Gradient Derivation (Backpropagation)
- Mathematically derived forward pass and backward pass computations for a custom 2-layer ANN with 2 inputs, 3 hidden neurons (ReLU), and 1 output neuron (sigmoid).
- Computed gradients using **Mean Squared Error (MSE)** loss function.
- Carried out weight updates for specific paths (e.g., `w7`, `w8`, `w9`) and interpreted the results.
- Analyzed **symmetry-breaking** issues when using identical weights and biases.
- Implemented conditional weight updates based on the parity of the student ID's last digit (as a design constraint).
- 📝 Work documented in `Katta_HW1.pdf` with all equations and flow written out by hand.

---

### 🔹 2. Binary Classification: MNIST Digits (5 vs 8)
- Filtered MNIST dataset to only include digits **5** and **8**, creating a balanced binary classification problem.
- Converted labels: 5 → 0, 8 → 1.
- Used a **feedforward ANN** with:
  - Input layer: 784 features (flattened 28×28 image)
  - Two hidden layers: 128 and 64 neurons (ReLU)
  - Output layer: 1 neuron (Sigmoid for binary output)
- Applied **MinMax normalization**, train-validation-test split (80/20/20), and **stratified sampling**.
- Integrated **EarlyStopping** with patience = 3, monitoring validation loss, and restoring best weights.
- Trained using `batch_size=1000` and `epochs=100`.
- Achieved:
  - ✅ **Test Accuracy:** 99.09%
  - 📉 **Test Loss:** 0.0252
- Visualizations:
  - 📈 Training vs Validation Accuracy & Loss
  - 📊 Confusion Matrix
  - 🖼️ Prediction samples (9 random images with predicted labels)

---

### 🔹 3. Experimental Comparison: Activation × Initializer
- Conducted an exhaustive comparison of **9 ANN configurations** by combining:
  - **Weight Initializers:** `RandomNormal`, `HeNormal`, `GlorotUniform (Xavier)`
  - **Activation Functions:** `ReLU`, `Sigmoid`, `Tanh`
- For each configuration:
  - Trained on the same filtered binary dataset (5 vs 8)
  - Used a **3-hidden-layer architecture**: 128 → 64 → 32 → 1
  - Monitored loss and accuracy on both training and validation sets
  - Captured **confusion matrix**, **precision**, **recall**, **F1-score**
- Summarized results in a detailed **results table**, highlighting:
  - Best accuracy (~99.3%) from `Normal + ReLU`
  - Sigmoid performed more steadily (robust to overfitting), but less accurately
  - ReLU led to fast learning but occasional instability
- 📊 Plots included: loss curves, accuracy curves, confusion matrices (per combo)

---

### 🔹 4. Multi-class Classification (Digits 0–9)
- Built a **multi-class ANN** for full 10-class classification (digits 0 to 9) on MNIST.
- Preprocessing:
  - Normalized pixel values to [0, 1]
  - One-hot encoded target labels
- Network Architecture:
  - Input layer: 784-dim flattened image
  - Hidden Layers: 128 → 64 → 32 neurons (all ReLU)
  - Output Layer: 10 neurons with **softmax** activation
- Loss Function: **Categorical Crossentropy**
- Optimizer: **Adam**
- Training Setup:
  - Early stopping with patience = 3
  - Batch size = 128, epochs = 100
  - Validation split = 20%
- Achieved:
  - 🧪 **Test Accuracy:** 97.46%
  - 🧾 Detailed classification report showing:
    - High precision and recall across most digits
    - Slightly lower recall on digit 5, and slightly lower F1 on digit 9
- 📉 Loss and Accuracy trends indicated good generalization without overfitting

---

### 🔹 5. Hyperparameter Grid Search: Batch Size vs Learning Rate
- Designed a systematic experiment across:
  - Batch Sizes: `4`, `16`, `32`, `64`
  - Learning Rates: `0.01`, `0.001`, `0.0001`, `0.00001`
- Total of **16 ANN models** trained (4×4 grid)
- Network Architecture: Compact ANN with Flatten → 32 → 16 → 10 (Softmax)
- Tracked:
  - Validation accuracy
  - Test accuracy
  - Convergence speed and stability
- Key Findings:
  - Low learning rates like `1e-5` led to slow convergence even with small batches
  - Larger batches (64) combined with smaller learning rates led to smoother generalization
  - ✅ **Best generalization**: `batch_size=16` with `learning_rate=0.001`
- 📈 Plotted: Test Accuracy vs (Batch Size / Learning Rate) ratio for optimal tuning insights


---

## 📈 What This Case Study Demonstrates

- Theoretical grounding in gradient descent and weight updates.
- Effective use of deep learning frameworks (Keras) for structured experimentation.
- Empirical model comparison and reproducible evaluation.
- Strong understanding of model generalization and training dynamics.
- Clean, modular code in Jupyter notebooks with visual interpretability.

---

## 🔖 Tags

`Deep Learning` `Neural Networks` `MNIST` `TensorFlow` `Keras` `Model Evaluation`  
`Backpropagation` `Activation Functions` `Weight Initialization` `Hyperparameter Tuning`
