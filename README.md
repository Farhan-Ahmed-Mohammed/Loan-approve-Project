# 🏦 Loan Approval Prediction System

## 🚀 Overview

This project predicts whether a loan application will be approved or rejected using machine learning.

It uses a complete end-to-end ML pipeline with feature engineering, preprocessing, and model optimization, and is deployed as a Flask web application for real-time predictions.

---

## 🔥 Key Highlights

* End-to-end ML pipeline using **Scikit-learn Pipeline & ColumnTransformer**
* Feature engineering (income ratios, EMI, financial indicators)
* Handles missing data, encoding, and scaling in a single workflow
* Model optimization with overfitting control
* Achieved **~81% test accuracy**
* Validated using **5-fold cross-validation**
* Deployed using **Flask web application**

---

## 🧠 Model & Approach

* Model: **Random Forest Classifier**
* Preprocessing:

  * OneHotEncoding for categorical features
  * StandardScaler for numerical features
  * Imputation for missing values
* Feature Engineering:

  * Total Income
  * Income-to-Loan Ratio
  * EMI-based features

---

## 📊 Performance

* Test Accuracy: **~81%**
* Balanced performance with good generalization
* Evaluated using:

  * Confusion Matrix
  * Classification Report
  * Cross-validation

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Flask
* Pickle

---

## 💻 Web Application

A Flask-based web app allows users to input applicant details and get real-time loan approval predictions.

---

## 📂 Project Structure

```
Loan-approve-Project/
│── app.py
│── loan_model.pkl
│── model.py / notebook
│── requirements.txt
│── templates/
│   ├── frontend.html
│   └── result.html
```

---

## ▶️ How to Run

```bash
git clone https://github.com/Farhan-Ahmed-Mohammed/Loan-approve-Project
cd Loan-approve-Project
pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:5000/

---

## 💡 Key Insights

* Credit history is the most important factor in loan approval
* Income-based features significantly improve prediction accuracy
* Pipeline-based approach ensures clean and scalable ML workflow

---

## 👨‍💻 Author

**Mohammed Farhan Ahmed**
AI / Machine Learning Enthusiast

GitHub: https://github.com/Farhan-Ahmed-Mohammed

---
