# 🧠 Medical Image Analysis with AI — Radiomics Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyRadiomics](https://img.shields.io/badge/PyRadiomics-Enabled-orange)](https://pyradiomics.readthedocs.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Supervised-green)](https://scikit-learn.org/)

---

## 🧭 Project Overview

This project focuses on **radiomics-based medical image analysis** to build and structure a robust dataset of radiomic features for supervised classification problems.

By extracting **quantitative imaging biomarkers**—such as intensity, texture, shape, and spatial descriptors—this project aims to convert qualitative medical images into structured, high-dimensional information for use in **machine learning** models capable of identifying and classifying pathological patterns.

---
## 🎯 Objectives

- 📊 **Standardize the processing pipeline** for medical images (e.g., DICOM or NRRD formats) using reproducible preprocessing and feature extraction steps based on PyRadiomics.  
- 🧩 **Extract and organize radiomic features** (first-order statistics, texture matrices such as GLCM, GLRLM, GLSZM, NGTDM, GLDM, and shape descriptors).  
- 🏷️ **Integrate metadata and clinical labels** to enable training of ML/DL classifiers for diagnostic support.  
- 🧠 **Evaluate the discriminative power** of radiomic biomarkers for non-invasive diagnosis, prognosis, or treatment response assessment.  
- 🧪 Ensure methodological **rigor and reproducibility** with proper feature standardization, selection strategies, and robust evaluation metrics.

---

## 🧰 Tech Stack & Libraries

- Python 3.8+  
- [PyRadiomics](https://pyradiomics.readthedocs.io/)  
- NumPy / Pandas  
- scikit-learn  
- OpenCV  
- Matplotlib / Seaborn

---

## ⚙️ Installation

You can use this project either in **Google Colab** or your **local environment**.

### 🪐 Option 1: Run on Google Colab
1. Upload the notebook (`.ipynb`) to Colab.  
2. Install the required libraries:
   ```bash
   !pip install pyradiomics numpy pandas scikit-learn opencv-python matplotlib seaborn
   ```
3. Run the cells directly — no additional configuration required.

### 💻 Option 2: Run Locally
1. Clone the repository:
  ```bash
  git clone https://github.com/your-username/your-repository-name.git
  cd your-repository-name
  ```

2. Install dependencies:
 ```bash
pip install pyradiomics numpy pandas scikit-learn opencv-python matplotlib seaborn
```

3. Open the notebook with Jupyter or VS Code and run the cells.

---

## 📥Dataset & Preprocessing

 - Compatible formats: DICOM (.dcm) and NRRD (.nrrd)
 - Automatic loading and preprocessing of images.
 - Radiomic feature extraction through PyRadiomics.
 - Structured tabular dataset creation for ML analysis.

---

## 🧠 Feature Extraction

Radiomic features extracted:

 - First-order statistics
 - GLCM (Gray Level Co-occurrence Matrix)
 - GLRLM (Gray Level Run Length Matrix)
 - GLSZM (Gray Level Size Zone Matrix)
 - NGTDM (Neighborhood Gray Tone Difference Matrix)
 - GLDM (Gray Level Dependence Matrix)
 - Shape descriptors

---

## 🤖 Machine Learning Pipeline

 - Preprocessing: Feature standardization & dimensionality reduction (PCA).
 - Modeling: Initial experiments with K-Nearest Neighbors (KNN).
 - Evaluation: Accuracy, precision, recall, F1-score, and confusion matrix.

---

## Example Code Snippet

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```
---

## 📊 Evaluation Metrics

 - ✅ Accuracy
 - 🧮 Precision
 - 📈 Recall
 - 🧠 F1-Score
 - 🔎 Confusion Matrix Visualization

---

## 📄 Project Author

👨‍💻 Juan Sebastian Peña Valderrama

 - Biomedical Engineer & Artificial Intelligence Specialist. 
 - Radiomics & Imaging Processing Enthusiast.
 - Medical Imaging Analysis Researcher.

📬 Contributions, suggestions, and pull requests are welcome!
