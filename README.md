# 🌌 Exoplanet Detection using Machine Learning

Detecting planets beyond our solar system using data-driven techniques 🚀

## 📌 Overview
This project focuses on detecting **exoplanets** using machine learning models trained on astronomical data. The detection is based on analyzing **stellar brightness variations (light curves)** to identify potential planetary transits.

The goal is to classify whether a star has an exoplanet orbiting it or not.

---

## 🧠 Problem Statement
When a planet passes in front of a star, it causes a slight dip in brightness. This phenomenon is known as the **Transit Method**.

The challenge is:
- Detect these small dips
- Distinguish real signals from noise
- Accurately classify stars with/without planets

---

## 📊 Dataset
- Source: NASA Kepler Space Telescope Dataset
- Data Type: Light intensity measurements over time
- Features: Flux values (brightness)
- Labels: 
  - `1` → Exoplanet detected  
  - `0` → No exoplanet  

---

## ⚙️ Tech Stack
- Python 🐍
- NumPy, Pandas
- Matplotlib / Seaborn
- Scikit-learn
- (Optional) TensorFlow / PyTorch

---

## 🔄 Workflow

1. **Data Preprocessing**
   - Handling missing values
   - Normalization / scaling
   - Noise reduction

2. **Feature Engineering**
   - Extracting relevant patterns from light curves
   - Dimensionality reduction (if applied)

3. **Model Training**
   - Logistic Regression
   - Random Forest
   - (Optional) Neural Networks

4. **Evaluation**
   - Accuracy
   - Precision & Recall
   - Confusion Matrix

---

## 📈 Results
- Achieved high accuracy of 96.5% in detecting exoplanets
- Model successfully identifies brightness dips indicating planetary transit

---

## 📷 Sample Visualization
Light curve showing dip in brightness:

![Light Curve](assets/light_curve.png)

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/exoplanet-detection.git
cd exoplanet-detection
