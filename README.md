#  ML Task 1: Linear and Logistic Regression from Scratch

##  Overview
This project implements **Linear Regression** and **Logistic Regression** entirely **from scratch** using only **NumPy** and **Pandas**, without using any pre-built machine learning models from libraries like scikit-learn.  

Both models use **Gradient Descent** optimization, **custom cost functions**, and include performance evaluation and visualization of learning curves.

---

##  Datasets Used

### 1. Linear Regression – Student Scores Dataset
- **Source:** [Student Scores Dataset (Kaggle/Open Data)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Description:** Predicts student exam scores based on hours studied.
- **Features:**
  - `Hours`: Number of study hours  
  - `Scores`: Marks obtained
- **Objective:** Build a simple regression model to predict scores using the number of study hours.

---

### 2. Logistic Regression – Heart Disease Dataset
- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Used CSV:** `https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv`
- **Description:** Classifies patients as having heart disease (`1`) or not (`0`).
- **Features:** Age, sex, cholesterol, blood pressure, max heart rate, chest pain type, etc.
- **Objective:** Predict the presence of heart disease based on patient attributes.

---

##  Implementation Details

###  Linear Regression
1. **Data Preprocessing:** Loaded and split data (80% train, 20% test).  
2. **Feature Scaling:** Standardized inputs for faster convergence.  
3. **Cost Function:**  
   - Used **Mean Squared Error (MSE)**  
     \[
     J(θ) = (1 / (2m)) * sum_{i=1..m} ( h_θ(x^(i)) - y^(i) )^2
     \]
4. **Gradient Descent:**  
   - Updated parameters iteratively to minimize MSE.
5. **Evaluation Metrics:**  
   - **MSE** and **R² Score**
6. **Visualization:**  
   - Cost vs Epoch curve  
   - Actual vs Predicted plot

---

###  Logistic Regression
1. **Data Preprocessing:** Loaded heart disease dataset, split into train/test.  
2. **Feature Scaling:** Standardized all numeric features.  
3. **Sigmoid Function:**  
   \[
   σ(z) = 1 / (1 + e^-z)
   \]

4. **Cost Function:**  
   - Used **Binary Cross-Entropy Loss**  
     \[
     J(θ) = -(1 / m) * sum_{i=1..m} [ y^(i) * log(h_θ(x^(i))) + (1 - y^(i)) * log(1 - h_θ(x^(i))) ]
     \]
5. **Gradient Descent:** Optimized parameters to minimize cost.  
6. **Evaluation Metrics:**  
   - **Accuracy** and **Confusion Matrix**
7. **Visualization:**  
   - Cost vs Epoch curve  
   - Confusion Matrix Heatmap

---

##  Evaluation Results

| Model | Metric | Train | Test |
|--------|--------|--------|--------|
| **Linear Regression** | MSE | ~10–15 | ~12–16 |
| **Linear Regression** | R² Score | ~0.94 | ~0.92 |
| **Logistic Regression** | Accuracy | ~0.86 | ~0.83 |
| **Logistic Regression** | Confusion Matrix | [[TN, FP], [FN, TP]] |

> *Results may vary slightly depending on random initialization.*

---

##  Key Concepts Implemented
- **Cost Function:** Quantifies model error during training.  
- **Gradient Descent:** Iteratively updates model parameters to minimize cost.  
- **Sigmoid Function:** Converts linear outputs into probabilities.  
- **Decision Boundary:** Separates data classes in logistic regression.  
- **Evaluation Metrics:** Used to assess model accuracy and performance.

---

##  Visual Outputs
- `linear_regression_loss.png` → Cost vs Iterations (MSE)
- `linear_regression_predictions.png` → Actual vs Predicted Student Scores
- `logistic_regression_loss.png` → Cost vs Iterations (Cross-Entropy)
- `logistic_regression_confusion_matrix.png` → Confusion Matrix (Heart Disease)

---

##  Requirements
```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
```
---

## References
- https://youtu.be/jGwO_UgTS7I?si=PzRwPbq6btNAtR8L - Andrew NG ML course(Linear and Logistic Regression)
- Kaggle and UCI - For datasets

---
**Author**: Giridhar Sreekumar<br>
**Date**: 3rd November 2025  
**Task**: ML Task 1 - Linear and Logistic Regression
