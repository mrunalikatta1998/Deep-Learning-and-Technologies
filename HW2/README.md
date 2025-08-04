# 🧠 Deep Learning Project: 3D Medical Imaging, Skin Lesion Classification & ResNet on CIFAR-10

Here I have applied a few deep learning techniques across diverse domains—3D medical imaging, skin disease classification, and general object recognition—using custom CNNs, pretrained models, Grad-CAM for explainability, and model compression through quantization. The project is organized into three major parts.

---

## 1. Rib Fracture Classification with 3D CNN (FractureMNIST3D)

### 🏁 Objective:
To build a 3D CNN model capable of classifying 3 types of rib fractures using volumetric CT scan images.

### 🧠 Dataset Details:
- Dataset: `FractureMNIST3D` from `medmnist`
- Input Shape: (28×28×28), grayscale CT cubes
- Classes: 0 – Buckle, 1 – Nondisplaced, 2 – Displaced
- Splits: official train/val/test split used

### 🧱 Step-by-Step Implementation:

1. **Dataset Preprocessing**:
   - Downloaded the dataset using `medmnist`.
   - Converted image tensors to numpy arrays.
   - Reshaped each input to `(28, 28, 28, 1)` and normalized to [0, 1].
   - One-hot encoded the labels for 3 classes.

2. **Model Architecture**:
   - Built a 3D CNN using `Conv3D → BatchNorm → MaxPooling3D` layers × 3.
   - Flattened the output and added:
     - Dense(128, ReLU) + Dropout(0.5)
     - Dense(64, ReLU) + Dropout(0.5)
     - Dense(3, Softmax)

3. **Training Strategy**:
   - Loss: Categorical Crossentropy
   - Optimizer: Adam with lr = 0.00022
   - Class imbalance handled using `class_weight`
   - EarlyStopping and ReduceLROnPlateau used to prevent overfitting and slow learning

4. **Evaluation**:
   - Accuracy, Precision, Recall, F1-score computed on test set
   - Confusion matrix plotted
   - Slice visualizations displayed with predicted vs. actual labels

### 📊 Result:
- Test Accuracy: 52.08%
- Weighted F1 Score: 0.5069
- Confusion matrix indicated moderate confusion between nondisplaced and displaced classes

---

## 2a. Skin Disease Classification using CNN + Augmentation (DermaMNIST)

### 🏁 Objective:
To build a 2D CNN model for classifying skin lesions into 7 disease categories, and enhance performance using data augmentation.

### 🧠 Dataset:
- Dataset: `DermaMNIST` (from `medmnist`)
- Input: RGB images resized to 28×28
- Classes: 7 diseases including melanoma, carcinoma, keratosis, etc.

### 🧱 Step-by-Step Implementation:

1. **Preprocessing**:
   - Downloaded data using `medmnist`.
   - Normalized images to [0, 1].
   - One-hot encoded the 7 class labels.

2. **Initial CNN Architecture**:
   - 4 Conv2D blocks:
     - Conv2D → BatchNorm → MaxPooling → Dropout (rates: 0.2, 0.3, 0.4)
     - Filters: 32 → 64 → 128 → 256
   - Fully Connected layers:
     - Dense(256, ReLU) → Dropout(0.4)
     - Dense(128, ReLU) → Dropout(0.3)
     - Dense(7, Softmax)

3. **Training Setup**:
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Epochs: 150 with EarlyStopping
   - Class weights computed for label imbalance

4. **Image Augmentation**:
   - Applied on-the-fly using `ImageDataGenerator`:
     - `rotation_range=15`, `zoom_range=0.2`, `shift_range=0.1`, `horizontal_flip=True`
   - Retrained same CNN on augmented data

5. **Evaluation**:
   - Compared models trained with and without augmentation
   - Plotted learning curves and confusion matrices
   - Visualized predictions on test images

### 📊 Results:
| Version              | Accuracy | F1 Score | Precision | Recall  |
|----------------------|----------|----------|-----------|---------|
| Without Augmentation | 53.87%   | 0.5830   | 0.6516    | 0.5387  |
| With Augmentation    | 74.17%   | 0.7194   | 0.7178    | 0.7417  |

---

## 2b. Transfer Learning using VGG-19 on DermaMNIST

### 🏁 Objective:
To leverage pretrained ImageNet weights using VGG-19 to improve classification accuracy on the same DermaMNIST task.

### 🧱 Implementation Steps:

1. **Resizing Input**:
   - DermaMNIST images resized from 28×28×3 to 32×32×3 to match VGG input shape

2. **Model Architecture**:
   - Base: VGG-19 (`include_top=False`)
   - Output taken from: `block4_conv1`
   - On top of that:
     - GlobalAveragePooling2D
     - Dense(128, ReLU)
     - Dense(7, Softmax)

3. **Training**:
   - All VGG layers frozen
   - Only the custom classifier was trained
   - EarlyStopping used to monitor validation loss

### 📊 Result:
- Test Accuracy: 64.09%
- Weighted F1 Score: 0.6676

### 🔍 Insight:
While the model used strong pretrained features, freezing all convolutional layers limited its adaptability. Performance was slightly below the custom CNN + augmentation approach.

---

## 2c. Grad-CAM Visualization

### 🏁 Objective:
To explain CNN predictions on DermaMNIST using heatmaps that show regions the model focused on.

### 🧱 Steps Taken:
1. Used trained CNN from 2a (with augmentation)
2. Applied Grad-CAM for test samples from each class
3. Generated overlayed heatmaps to highlight feature activations
4. Displayed predicted and actual class labels

### ✅ Insight:
Grad-CAM provided visual justification for predictions, showing high activation around lesion areas—building trust in model behavior.

---

## 2d. Model Quantization for Edge Deployment

### 🏁 Objective:
To reduce model size and computational cost using three types of quantization techniques.

### 🧱 Quantization Methods:
1. **Post-training Dynamic Range Quantization**
2. **Full Integer Quantization**
3. **Quantization-Aware Training (QAT)**

### 🧪 Evaluation Metrics:
- Test Accuracy
- Model Size
- Training vs deployment complexity

### 📊 Results:

| Method                    | Accuracy | Size Reduction | Notes                              |
|---------------------------|----------|----------------|------------------------------------|
| Post-training (Dynamic)   | ~72%     | Medium          | Simple, fast, suitable for mobile  |
| Full Integer Quantization | ~70%     | High            | Best size vs performance tradeoff  |
| QAT                       | ~74%     | Moderate        | Best accuracy, needs retraining    |

---

## 3. CIFAR-10 Classification with Custom ResNet

### 🏁 Objective:
To implement a ResNet-style architecture from scratch for general-purpose image classification.

### 🧠 Dataset:
- CIFAR-10: 60,000 images (32×32×3), 10 classes
- Used standard 50K train / 10K test split

### 🧱 Architecture:

1. **Initial Layer**:
   - Conv2D (filters=32) → BatchNorm → ReLU

2. **Section A**: 3 × Residual Blocks with 32 filters  
3. **Section B**: 3 × Residual Blocks with 64 filters  
4. **Section C**: 3 × Residual Blocks with 128 filters  

Each Residual Block:
- Conv → BatchNorm → ReLU → Conv → BatchNorm → Skip connection → ReLU

5. **Output Head**:
   - Global Average Pooling
   - Flatten → Dense(10, softmax)

### ⚙️ Training:
- Epochs: 20+
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Monitored validation accuracy & loss

### 📊 Results:
- Test Accuracy: **77.55%**
- Test Loss: **0.71717**
- F1 Score: **0.75952**
- Precision: **0.80976**
- Recall: **0.72220**

---

## 🧾 Final Summary Table

| Task                     | Model Type             | Accuracy  | F1 Score | Precision | Recall  | Dataset         |
|--------------------------|------------------------|-----------|----------|-----------|---------|-----------------|
| Fracture Classification  | 3D CNN                 | 52.08%    | 0.5069   | 0.5013    | 0.5208  | FractureMNIST3D |
| Skin Lesion (baseline)   | CNN (no aug)           | 53.87%    | 0.5830   | 0.6516    | 0.5387  | DermaMNIST      |
| Skin Lesion (augmented)  | CNN + Augmentation     | 74.17%    | 0.7194   | 0.7178    | 0.7417  | DermaMNIST      |
| Transfer Learning        | VGG-19 (frozen layers) | 64.09%    | 0.6676   | 0.7141    | 0.6409  | DermaMNIST      |
| Quantized Model (QAT)    | CNN + QAT              | ~74%      | ~0.72    | ~0.72     | ~0.74   | DermaMNIST      |
| CIFAR-10 Classification  | Custom ResNet          | 77.55%    | 0.7595   | 0.8098    | 0.7222  | CIFAR-10        |

---

## 🧰 Tools & Libraries

- **Python 3.10**
- **TensorFlow / Keras**
- **NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn**
- **OpenCV, medmnist**
- **Jupyter Notebook (Google Colab)**

---

## 📌 Tags

`CNN` `3D Vision` `Medical Imaging` `Skin Disease Classification` `Transfer Learning`  
`Grad-CAM` `Model Quantization` `Residual Networks` `Image Classification` `CIFAR10`
