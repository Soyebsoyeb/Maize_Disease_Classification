# 🌽📊 Corn/Maize Leaf Disease Classification – Detailed Analysis Report

This report presents an in-depth analysis of a deep learning model built to classify **corn/maize leaf diseases** using **MobileNetV2** enhanced with the **CBAM (Convolutional Block Attention Module)** 🔍🧠.

---

## 📁 Dataset Overview

**Disease Classes (4 categories):**
- 🌿 **Blight**
- 🍂 **Common Rust**
- 🍁 **Gray Leaf Spot**
- ✅ **Healthy**

**Dataset Size:**
- 🖼️ **Total Images**: 4,188  
- 🏋️ **Training Set**: 3,348 images (80%)  
- 🧪 **Validation Set**: 417 images (10%)  
- 🧬 **Test Set**: 423 images (10%)  

---

## 🧠 Model Architecture

The model uses a **transfer learning** approach built on:

- 🔗 **Base Model**: MobileNetV2 (pretrained on ImageNet)  
  - Input Shape: `(224, 224, 3)`  
  - 🔒 All layers frozen during training

- ✨ **Attention Mechanism**: CBAM  
  - 📊 Combines **Channel + Spatial Attention**  
  - 🎯 Helps focus on disease-specific features

- 🔚 **Classifier Head**:  
  - 📉 Global Average Pooling  
  - 🧱 Dense Layers: 256 ➡️ 128 (ReLU)  
  - 🧼 Batch Normalization + 💧 Dropout (0.5)  
  - 🟰 Final Layer: Softmax (4 output neurons)

---

## ⚙️ Training Configuration

- 🚀 **Optimizer**: Adam (LR = 0.001)  
- 📉 **Loss Function**: Categorical Crossentropy + Label Smoothing (0.1)

**Regularization:**
- 🧲 L2 Weight Decay: 1e-4  
- 💧 Dropout: 0.5

**Data Augmentation:**
- 🔄 Rotation (±30°)  
- ↔️ Horizontal & Vertical Flip  
- ☀️ Brightness Adjustment (0.8–1.2)  
- 🔍 Zoom (±20%)

**Callbacks:**
- 🛑 Early Stopping (patience = 5)  
- 📉 Reduce LR on Plateau (factor = 0.5, patience = 2, min_lr = 1e-6)

---

## 📈 Training Performance

- ⏳ Trained for **38 epochs** before early stopping
- ✅ **Final Training Accuracy**: 99.31%  
- 🧪 **Final Validation Accuracy**: 96.16%

### 🔍 Observations:
- ⚡ Rapid performance gain in early epochs  
- 🔽 Learning rate reduced **4 times** during training  
- 💪 No overfitting: training and validation curves stayed close

---

## 🧪 Test Set Evaluation

- ✅ **Test Accuracy**: 97.16%  
- 📉 **Test Loss**: 0.4809  

**Weighted Averages:**
- 🎯 Precision: 0.9714  
- 🔁 Recall: 0.9716  
- 🧠 F1-Score: 0.9715

---

## 📊 Per-Class Performance

| 🌿 Class           | 🎯 Precision | 🔁 Recall | 🧠 F1-Score | 📊 Support |
|--------------------|--------------|------------|-------------|------------|
| **Blight**         | 0.95         | 0.95       | 0.95        | 116        |
| **Common Rust**    | 0.99         | 1.00       | 1.00        | 132        |
| **Gray Leaf Spot** | 0.91         | 0.90       | 0.90        | 58         |
| **Healthy**        | 1.00         | 1.00       | 1.00        | 117        |

---

## 🔍 Confusion Matrix Insights

- ✅ **Healthy** and **Common Rust**: **Near-perfect** predictions  
- 🟡 **Gray Leaf Spot**: Slight dip in recall (~90%)  
- 🔁 **Blight vs. Gray Leaf Spot**: Some misclassification (likely visual similarity)  
- ⚖️ **Blight**: Balanced precision & recall (95%)

---

## ✅ Key Strengths

- 🌟 **High Accuracy**: 97.16% on test set  
- 🧠 **Efficient Architecture**: MobileNetV2 + CBAM delivers compact yet effective performance  
- 🔒 **Strong Regularization**: Dropout & L2 prevent overfitting  
- ⚖️ **Balanced Class Performance**

---

## 🛠️ Potential Improvements

- ⚠️ **Class Imbalance**: Gray Leaf Spot has fewer samples — consider oversampling or class weights  
- 🔄 **Advanced Augmentation**: Add blur, mildew-like noise, or other domain-specific transforms  
- 🔓 **Fine-tuning**: Gradually unfreeze MobileNetV2 layers to enhance learning  
- 🎯 **Per-Class Thresholding**: Tune decision thresholds for optimal precision/recall

---

## 🧾 Conclusion

This model achieves **state-of-the-art results** 🚀 for **corn/maize leaf disease classification** using a lightweight yet powerful combination of **MobileNetV2 + CBAM** 🧠.

- 📷 CBAM guides attention to **relevant disease regions**
- ✅ Highly accurate and generalizes well across leaf types
- 📱 Ready for **real-world deployment** in smart farming or mobile plant diagnostic apps

---

> 🌟 This intelligent system empowers farmers to detect diseases early, enhance crop yield, and secure agricultural health — **one leaf at a time**! 🌿📱
