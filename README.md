# 🎯 Smart Loan Approval System – Stacking Ensemble Model

This project implements a **Smart Loan Approval System** using a **Stacking Ensemble Machine Learning model** to predict whether a loan will be approved or rejected based on applicant details.

The system combines multiple machine learning models and provides a clean, user-friendly web interface built with **Streamlit**.

---

## 🚀 Project Overview

Loan approval is a critical decision-making process for financial institutions.  
Instead of relying on a single model, this system uses **ensemble learning (stacking)** to improve robustness and decision quality.

### Key Highlights
- Uses **Stacking Ensemble Learning**
- Combines predictions from multiple base models
- Interactive web interface using Streamlit
- Provides business-friendly explanations for predictions

---

## 🧠 Machine Learning Approach

### 🔹 Base Models
- Logistic Regression  
- Decision Tree  
- Random Forest  

### 🔹 Meta Model
- Logistic Regression  

The base models generate predictions, which are then used as input features for the meta-model.  
This allows the system to **learn how to optimally combine model outputs** rather than relying on a single algorithm.

---

## 🖥️ Web Application (UI)

The application allows users to input applicant details such as:
- Applicant Income  
- Co-Applicant Income  
- Loan Amount  
- Loan Term (Years)  
- Credit History  
- Employment Status  
- Property Area  

The system predicts:
- **Loan Approved / Loan Rejected**
- Confidence score
- Clear business-level explanation of the decision

---

## 📸 Application Screenshot

Below is a screenshot of the running application:

![Smart Loan Approval System](https://github.com/user-attachments/assets/d2fa38ec-ba9b-4601-a5f0-7b5c1d18f64e)

---

---

## ⚙️ Technologies Used

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Streamlit  
- Joblib  

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AshrithaShaganti/Stacking-Ensemble-Loan-Approval.git
cd Stacking-Ensemble-Loan-Approval
```
### 2️⃣Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run the Application
streamlit run app.py

### 📊 Model Evaluation Summary
Individual models were evaluated using accuracy and confusion matrix.

Logistic Regression performed best among individual models.

The stacking model demonstrates how ensemble techniques combine multiple models, though stacking does not always outperform the best individual model.

This highlights the importance of choosing models based on problem complexity and data characteristics.


### 📌 Key Learning Outcomes

Practical implementation of stacking ensemble models

Importance of proper preprocessing and feature scaling

Understanding trade-offs between model complexity and performance

End-to-end ML workflow: training → evaluation → deployment



