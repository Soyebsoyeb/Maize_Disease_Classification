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




