# 💰 Loan Approval Prediction using Machine Learning

## 📌 Project Overview
This project predicts whether a loan application will be approved or rejected based on applicant information such as income, credit history, education, dependents, and property area.

The system uses **Machine Learning with Scikit-learn** and implements a **Random Forest Classifier** inside a pipeline that performs both preprocessing and model training.

The goal of this project is to automate loan approval prediction and assist banks or financial institutions in decision-making.

---

## 📂 Dataset

Dataset used: **Loan Data Dataset**

The dataset contains information about loan applicants.

Columns in the dataset:

- Loan_ID – Unique loan identifier  
- Gender – Applicant gender  
- Married – Marital status  
- Dependents – Number of dependents  
- Education – Graduate / Not Graduate  
- Self_Employed – Self employment status  
- ApplicantIncome – Applicant income  
- CoapplicantIncome – Co-applicant income  
- LoanAmount – Requested loan amount  
- Loan_Amount_Term – Loan repayment period  
- Credit_History – Credit history of the applicant  
- Property_Area – Urban / Rural / Semiurban  
- Loan_Status – Target variable (Loan approval status)

Target values:

- **Y → Loan Approved**
- **N → Loan Rejected**

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Pickle

---

## 🔄 Data Preprocessing

Several preprocessing steps were applied before training the model.

### Handling Categorical Data
Categorical columns were converted to numerical values using:

OneHotEncoder

This avoids assigning incorrect numerical rankings to categorical values like Yes/No.

### Feature Scaling
Numerical features were standardized using:

StandardScaler

This ensures all numerical features are on the same scale.

### Handling Missing Values
Missing values in numerical columns were filled using the **mean of the training data** to prevent data leakage.

---

## 🔧 Machine Learning Pipeline

A **Scikit-learn Pipeline** was used to combine preprocessing and model training.

Pipeline steps:

1. ColumnTransformer  
2. OneHotEncoder for categorical features  
3. StandardScaler for numerical features  
4. RandomForestClassifier for prediction  

Using a pipeline ensures that preprocessing and training happen **in one integrated workflow**.

---

## 🤖 Model Used

Model: **Random Forest Classifier**

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## 📊 Model Evaluation

The model was evaluated on the test dataset.

Accuracy:

0.7886 (≈ 79%)

Confusion Matrix:

```
[[18 25]
 [ 1 79]]
```

Classification Report:

Class N  
Precision: 0.95  
Recall: 0.42  
F1-score: 0.58  

Class Y  
Precision: 0.76  
Recall: 0.99  
F1-score: 0.86  

Overall accuracy of the model is approximately **79%**.

---

## 🔮 Prediction Example

Example prediction using a sample loan application:

Applicant details:

- Gender: Male  
- Married: Yes  
- Dependents: 1  
- Education: Graduate  
- Self Employed: No  
- Applicant Income: 4500  
- Coapplicant Income: 3000  
- Loan Amount: 128  
- Loan Term: 360  
- Credit History: 1  
- Property Area: Urban  

Prediction:

Loan Approved → **Y**

Probability of approval:

**0.81 (81%)**

---

## 💾 Model Saving

The trained model is saved using **Pickle** so it can be reused later without retraining.

Saved file:

loan_model.pkl

Example code used to save the model:

```python
with open("loan_model.pkl","wb") as f:
    pickle.dump(model,f)
```

---

## 💡 Applications

This system can be used for:

- Bank loan approval automation
- Financial risk assessment
- Loan eligibility prediction
- FinTech platforms

---

## 🔮 Future Improvements

Possible improvements include:

- Hyperparameter tuning for better accuracy
- Using advanced models like XGBoost or Gradient Boosting
- Deploying the model as a web application using Flask
- Creating a dashboard for loan approval prediction

---

## 👨‍💻 Author

Farhan Ahmed Mohammad  
AI / Machine Learning Enthusiast
