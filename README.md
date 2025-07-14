Dataset 1 : /kaggle/input/corn-or-maize-leaf-disease-dataset/data

# 🌽 Maize Leaf Disease Classification

---

## 📁 Dataset Info

- **Dataset Path**: `/kaggle/input/corn-or-maize-leaf-disease-dataset/data`
- **Detected Classes**:
  - 🌱 Blight
  - 🍂 Common Rust
  - 🍁 Gray Leaf Spot
  - ✅ Healthy

### 🔄 Dataset Split

| Split        | Percentage |
|--------------|------------|
| 🏋️‍♂️ Training   | 80%        |
| 🧪 Validation | 10%        |
| 🧾 Test       | 10%        |

---

## 🤖 Project Objective

This program classifies maize leaf images into the above four categories using:

- 📷 **CNN with MobileNetV2**
- 🧠 **CBAM (Convolutional Block Attention Module)** for better feature attention
- 💡 **Support Vector Machine (SVM)** on extracted features to enhance classification performance

---

## 🧠 CBAM Attention Module

CBAM helps the network **focus on important regions** of the image.

### 📌 Structure

- 🔴 **Channel Attention**  
  `GlobalAvgPool + GlobalMaxPool → MLP → Add + Sigmoid`

- 🔵 **Spatial Attention**  
  `Channel-wise mean/max → Concat → Conv2D → Sigmoid`

> ✅ This is implemented in the function `cbam_block(input_feature)`

---

## 🧬 Full Architecture

Input → MobileNetV2 → CBAM → GAP → Dense(256) → BN → Dropout(0.5)
↓
Dense(128) → BN → Dropout(0.5)
↓
Dense(4, softmax)


---

## 🏋️‍♂️ Training Results

| Metric      | Value   |
|-------------|---------|
| 🎯 Accuracy | 97.16%  |
| 📉 Final Loss | 0.4247 |
| ⏱️ Epochs   | ~41     |

📈 **Training Graph**: Accuracy and loss plotted across epochs.

---


## 📊 Evaluation (CNN Output)

| Class           | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| 🌱 Blight         | 0.96      | 0.93   | 0.95     | 116     |
| 🍂 Common_Rust    | 1.00      | 1.00   | 1.00     | 132     |
| 🍁 Gray_Leaf_Spot | 0.87      | 0.93   | 0.90     | 58      |
| ✅ Healthy        | 1.00      | 1.00   | 1.00     | 117     |
| **Weighted Avg**  | **0.97**  | **0.97** | **0.97** | **423** |



📊 **Confusion Matrix**: Displayed using Seaborn heatmap.

---

## ✅ Final Summary

| 🔍 Component           | 🔢 Result/Value         |
|------------------------|-------------------------|
| 🎯 Deep Model Accuracy | 97.16%                  |
| 🧠 SVM Accuracy         | 98.11%                  |
| 🧩 Base Architecture    | MobileNetV2 + CBAM      |
| 📐 Feature Dimension    | 128                     |
| 🏋️ Training Images      | 3348                    |
| 🧾 Test Images          | 423                     |
| 🏆 Best Classifier      | SVM on features         |
| ❌ Misclassifications   | Minimal (mostly in Gray Leaf Spot) |





Dataset 2: /kaggle/input/maizeleaf/MaizeLeafDataset

# 🌽 Maize Leaf Disease Classification – Advanced (5-Class)

---

##  1. Dataset and Preprocessing

### 📂 Dataset Structure
- **Source**: `/kaggle/input/maizeleaf/MaizeLeafDataset`
- **Classes**:
  - 🍂 Common Rust  
  - 🍁 Gray Leaf Spot  
  - ✅ Healthy  
  - 🌫️ Northern Leaf Blight  
  - ❌ Not Maize Leaf

### 📸 Sample Visualization
- 4 random images per class displayed using `matplotlib` to ensure data integrity.

### 🔀 Dataset Splitting

| Split       | Percentage | Destination Folder                         |
|-------------|------------|--------------------------------------------|
| 🏋️‍♂️ Train     | 80%        | `/kaggle/working/split_data/train/`        |
| 🧪 Validation | 10%        | `/kaggle/working/split_data/val/`          |
| 🧾 Test       | 10%        | `/kaggle/working/split_data/test/`         |

---

##  2. Image Data Generators (Augmentation)

- **Training Generator** includes:
  - `rescale`, `rotation_range`, `horizontal_flip`, `vertical_flip`, `brightness_range`, `zoom_range`, `fill_mode`

- **Validation & Test**:
  - Only `rescale`

- Output Shape: `(224, 224, 3)`
- Labels: One-hot encoded

---

##  3. 🧠 CBAM Attention Mechanism

The **Convolutional Block Attention Module (CBAM)** improves the model’s ability to focus on relevant leaf regions.

### 🔍 Module Structure:
- 🔴 **Channel Attention**:  
  `GlobalAvgPool + GlobalMaxPool → MLP → Add → Sigmoid`
- 🔵 **Spatial Attention**:  
  `Mean + Max → Concatenate → Conv2D → Sigmoid`

> 💡 Integrated via `cbam_block(input_feature)` into the main model.

---

## 4. 🏗 Model Architecture

### 📐 Structure:

Input → MobileNetV2 → CBAM → GAP → Dense(256) → BN → Dropout(0.5)
↓
Dense(128) → BN → Dropout(0.5)
↓
Dense(5, softmax)



- 🧠 **Backbone**: MobileNetV2 (`include_top=False`, pretrained on ImageNet)
- 🔧 **Trainable Parameters**: ~3M
- 🧪 **Activation**: `softmax`
- 📉 **Loss**: `CategoricalCrossentropy` (with `label_smoothing=0.1`)
- ⚙️ **Optimizer**: Adam (`lr=0.001`)

### 🧬 Callbacks:
- `EarlyStopping(patience=5)`
- `ReduceLROnPlateau`

---

## 5. 🏋️ Training Pipeline

- 📈 **Training completed in ~45 epochs**  
- ✅ **Final Validation Accuracy**: ~99.66%  
- 💾 Model saved as: `final_trained_model.h5`

---

## 6. 📊 Evaluation (CNN)

### 📉 Test Results:
- **Loss**: ~0.43  
- **Accuracy**: ~99.00%

### 📌 Classification Metrics:

| Metric              | Value    |
|---------------------|----------|
| 🎯 Precision (weighted) | 0.9903 |
| 🔁 Recall (weighted)    | 0.9899 |
| 🧠 F1-Score (weighted)  | 0.9899 |

### 🔍 Confusion Matrix:
- Visualized using Seaborn heatmap
- Minimal misclassification across all 5 classes

---

## ✅ 7. 📦 Feature Extraction (CNN + SVM Hybrid)

- 🧬 **Output**: 128-dimensional feature vector per image (from CBAM-enhanced MobileNetV2)

### 📌 Feature Shapes

| Set        | Shape       |
|------------|-------------|
| 🏋️ Train   | (7079, 128) |
| 🧾 Test    | (891, 128)  |

---



## 📈 Performance

| 🧪 Metric              | 📊 Value              |
|------------------------|-----------------------|
| 🎯 Test Accuracy       | **99.44%**            |
| 🧠 F1, Precision, Recall | **> 0.98** for all classes |

> 🚀 **SVM slightly outperforms the CNN softmax classifier!**

---

## ✅ Conclusion & Summary

| 🧩 Component         | 💡 Description                                           |
|---------------------|----------------------------------------------------------|
| 🏗 Model Backbone    | MobileNetV2 (ImageNet pre-trained) + CBAM                |
| 🧠 Classification    | Softmax and SVM (on extracted 128-dim features)          |
| 🖼️ Input Size        | 224x224 RGB                                              |
| 🔄 Data Augmentation | ✅ Applied to training set                               |
| 🔎 CBAM              | ✅ Channel + Spatial attention                           |
| 🎯 Final Accuracy    | ~**99.4%** (combined CNN + SVM performance)              |
| 🌽 Use Case          | Maize Leaf Disease Classification                        |
| 🚀 Deployment Ready  | ✅ Saved model (`.h5`) and feature extractor exported    |






