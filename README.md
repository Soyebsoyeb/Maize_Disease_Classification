🌽📊 Corn/Maize Leaf Disease Classification – Detailed Analysis Report
This report provides an insightful breakdown of the deep learning model built to classify corn/maize leaf diseases using MobileNetV2 enhanced with a CBAM (Convolutional Block Attention Module) 🔍🧠.

📁 Dataset Overview
Classes (4 disease categories):

🌿 Blight

🍂 Common Rust

🍁 Gray Leaf Spot

✅ Healthy

Dataset Size:

🖼️ Total Images: 4,188

🏋️ Training Set: 3,348 images (80%)

🧪 Validation Set: 417 images (10%)

🧬 Test Set: 423 images (10%)

🧠 Model Architecture
This model uses a transfer learning approach with powerful components:

🔗 Base Model: MobileNetV2 (pretrained on ImageNet)

Input Shape: (224, 224, 3)

🔒 All layers frozen during training

✨ Attention Mechanism: CBAM

📊 Combines Channel + Spatial Attention

🎯 Focuses on important leaf features (disease signs)

🔚 Classifier Head:

📉 Global Average Pooling

🧱 Dense Layers: 256 ➡️ 128 (ReLU)

🧼 Batch Normalization + 💧 Dropout (0.5)

🟰 Final Layer: Softmax with 4 outputs

⚙️ Training Configuration
🚀 Optimizer: Adam (LR = 0.001)

📉 Loss Function: Categorical Crossentropy + Label Smoothing (0.1)

🧰 Regularization:

🧲 L2 Weight Decay: 1e-4

💧 Dropout: 0.5

🎛️ Data Augmentation:

🔄 Rotation (±30°)

↔️ Flip (horizontal/vertical)

☀️ Brightness: 0.8–1.2

🔍 Zoom: ±20%

⏱️ Callbacks:

🛑 Early Stopping (patience=5)

📉 Reduce LR on Plateau (factor=0.5, patience=2, min_lr=1e-6)

📈 Training Performance
Trained for 38 epochs before early stopping.

✅ Final Training Accuracy: 99.31%

🧪 Final Validation Accuracy: 96.16%

🔍 Key Observations:
⚡ Fast accuracy gains in early epochs

📉 Learning rate dropped 4️⃣ times

💪 No overfitting: Training and validation curves closely matched

🧪 Test Set Evaluation
✅ Test Accuracy: 97.16%

📉 Test Loss: 0.4809

🧮 Weighted Averages:
🎯 Precision: 0.9714

🔁 Recall: 0.9716

🧠 F1-score: 0.9715

📊 Per-Class Performance
🌿 Class	🎯 Precision	🔁 Recall	🧠 F1-score	📊 Support
Blight	0.95	0.95	0.95	116
Common Rust	0.99	1.00	1.00	132
Gray Leaf Spot	0.91	0.90	0.90	58
Healthy	1.00	1.00	1.00	117

🔍 Confusion Matrix Analysis
✅ Healthy and Common Rust: Near-perfect predictions

🟡 Gray Leaf Spot: Slight dip with 90% recall

🔁 Blight vs. Gray Leaf Spot: Most confusion observed (visual similarity)

⚖️ Blight maintains balanced precision & recall (95%)

✅ Key Strengths
🌟 High Accuracy: 97.16% is excellent for a 4-class classification

🧠 Strong Architecture: MobileNetV2 + CBAM delivers efficient, focused feature learning

🔒 Good Regularization: Dropout + L2 prevents overfitting

⚖️ Balanced Performance: No severe bias toward any class

🛠️ Potential Improvements
⚠️ Class Imbalance: Gray Leaf Spot has fewer samples — consider class-weighting or oversampling

🔄 More Targeted Augmentation: Add leaf-specific transforms (e.g., blur, mildew patterns)

🔓 Unfreeze Base Layers: Enable fine-tuning of MobileNetV2 to boost feature learning

🎯 Class-Specific Thresholding: Optimize decision boundaries per class to reduce false positives

🧾 Conclusion
The model achieves state-of-the-art performance 🚀 for classifying corn leaf diseases using a lightweight yet powerful combination of MobileNetV2 + CBAM 🧠.

📷 CBAM ensures the model pays attention to key disease patterns

✅ Accurate and generalizes well across diverse leaf conditions

🧑‍🌾 Highly suitable for real-time agricultural applications like mobile-based diagnosis or smart farming tools

🌟 This intelligent system can help farmers detect diseases early, boost yield, and protect crops—one leaf at a time! 🌿📱
