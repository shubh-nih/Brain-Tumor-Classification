# 🧠 Brain Tumor Classification using Deep Learning

A deep learning project that classifies brain MRI scans into 4 categories  **Glioma, Meningioma, No Tumor, and Pituitary**  using transfer learning with ResNet152, VGG19, and MobileNetV2.

---

## 📁 Project Structure

```
Brain-Tumor-Classification/
│
├── 01_Image_preprocessing.ipynb     # Raw MRI preprocessing pipeline
├── 2_model_selection.ipynb          # Baseline CNN + 3 transfer learning models
├── 3_fine_tuning_resnet152.ipynb     # Fine-tuned ResNet152 (best model)
└── README.md
```

---

## 📊 Dataset

- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Training images:** 5,600 (balanced 1,400 per class)
- **Test images:** 1,600 (400 per class)
- **Image size:** 224×224 pixels
- **Source:** [Kaggle Brain MRI Tumor Dataset]([https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset])

---

## 🔧 Preprocessing Pipeline (`01_Image_preprocessing.ipynb`)

Raw MRI images go through a custom OpenCV pipeline before training:

1. **Grayscale conversion** - removes color noise irrelevant to MRI
2. **Gaussian blur** - smooths high-frequency noise
3. **Otsu thresholding** - adaptive binary segmentation
4. **Contour detection** - finds the largest brain region
5. **Crop to ROI** - crops to bounding box of detected contours
6. **Resize to 224×224** - standardized input size
7. **CLAHE enhancement** - contrast-limited adaptive histogram equalization for better feature visibility

Preprocessed images were saved locally and later uploaded to Kaggle as a dataset for model training.

---

## 🤖 Model Selection (`2_model_selection.ipynb`)

### Baseline CNN
A custom 4 block CNN (Conv2D → BatchNorm → MaxPool) was trained first. It **failed to learn** (~25% accuracy, near random), confirming the need for transfer learning.

### Transfer Learning — 3 Models Compared

| Model | Test Accuracy | Test Loss | Params |
|---|---|---|---|
| **ResNet152** | **92.12%** | 0.4197 | ~110M |
| MobileNetV2 | 88.06% | 0.4526 | ~2.9M |
| VGG19 | 82.63% | 0.4982 | ~20M |

All models used:
- **ImageNet pretrained weights**, base layers frozen
- **GlobalAveragePooling2D** → Dense(512) → Dropout(0.5) → Dense(4, softmax)
- Trained for **15 epochs** with Adam optimizer
- **ResNet152** used `preprocess_input` from `tensorflow.keras.applications.resnet`

**ResNet152 was selected** for fine tuning due to highest accuracy and most stable training curves.

---

## 🎯 Fine-Tuning ResNet152 (`3_fine_tuning_resnet152.ipynb`)

### Strategy
- **Unfroze last 50 layers** of ResNet152 backbone
- Added a deeper classifier head: Dense(512) → BN → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(4)
- Trained up to **50 epochs** with early stopping

### Callbacks
- `EarlyStopping` - patience = 6 on `val_accuracy`, restores best weights
- `ModelCheckpoint` - saves best model (`.keras` format)
- `ReduceLROnPlateau` - reduces LR by 0.2x if `val_loss` plateaus for 3 epochs

### Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **95.06%** |
| Test Loss | 0.4306 |
| Macro Avg F1-Score | 0.95 |
| Total Misclassified | 79 / 1600 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Glioma | 0.99 | 0.83 | 0.90 |
| Meningioma | 0.91 | 0.98 | 0.94 |
| No Tumor | 0.92 | 1.00 | 0.96 |
| Pituitary | 0.99 | 0.99 | 0.99 |

> ⚠️ **Note:** Glioma recall (0.83) is the weakest point - the model misses ~17% of actual glioma cases. This is a known concern and will be addressed in future iterations.

---

## 🛠️ Tech Stack

- **Python** - NumPy, Pandas, Matplotlib, Seaborn
- **OpenCV** - Image preprocessing
- **TensorFlow / Keras** - Model building and training
- **Scikit-learn** - Evaluation metrics
- **GPU** - Tesla T4 (Kaggle, dual GPU)

---

## 👤 Author

**Shubham Bisht**
- LinkedIn: [Shubham Bisht](https://www.linkedin.com/in/shubhambisht7/)
