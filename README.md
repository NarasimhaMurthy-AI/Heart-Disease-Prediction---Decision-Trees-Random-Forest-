# Heart Disease Prediction - Decision Trees & Random Forest (Task 5)

## 📌 Overview
This project is part of **Task 5** in a machine learning series.  
We use the **Heart Disease dataset** to build and evaluate **Decision Tree** and **Random Forest** models, analyze model accuracy, visualize feature importances, and compare ROC curves.

The project covers:
- Data loading and preparation
- Splitting into training and test sets
- Training Decision Tree (full depth and limited depth)
- Accuracy vs. Max Depth analysis
- Training Random Forest
- Feature importance ranking
- ROC curve plotting
- Model accuracy comparison

---

## 📂 Dataset
- **File:** `heart.csv`
- **Target variable:** `target` (1 = Disease, 0 = No Disease)
- Features include age, sex, cholesterol levels, blood pressure, etc.

---

## ⚙️ Technologies Used
- Python 3
- Pandas, NumPy
- Matplotlib
- Scikit-learn

---

## 📊 Project Workflow
1. **Load Data**
   - Read CSV into pandas DataFrame
   - Separate features (X) and target (y)

2. **Train-Test Split**
   - 80% train / 20% test split
   - Stratified to preserve class balance

3. **Decision Tree (Full Depth)**
   - Train a full decision tree classifier
   - Plot and interpret the structure

4. **Accuracy vs Max Depth**
   - Loop over `max_depth` from 1 to 12
   - Plot training, testing, and cross-validation accuracy

5. **Decision Tree (Depth=10)**
   - Train a shallower tree
   - Visualize to see simplified rules

6. **Random Forest**
   - Train an ensemble of 200 trees
   - Rank top 10 most important features

7. **ROC Curves**
   - Plot ROC curves for all models
   - Calculate AUC scores

8. **Model Summary**
   - Print train/test accuracy for each model

---

## 📈 Outputs & Visuals
The script displays:
- Decision Tree (Full Depth)
- Accuracy vs Max Depth plot
- Decision Tree (Max Depth=10)
- Top 10 Feature Importances
- ROC Curves
- Printed model accuracy summary

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-task5.git
   cd heart-disease-task5
