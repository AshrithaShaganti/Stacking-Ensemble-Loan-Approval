import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="centered"
)

# -----------------------------
# Load CSS
# -----------------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------
# Load models
# -----------------------------
stack_model = joblib.load("models/stack_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1 class='title'>🎯 Smart Loan Approval System – Stacking Model</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Predict loan approval using a Stacking Ensemble Machine Learning model.</p>",
    unsafe_allow_html=True
)

# -----------------------------
# Applicant Details Card
# -----------------------------
st.markdown("<h2>📝 Applicant Details</h2>", unsafe_allow_html=True)

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Co-Applicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
        loan_term_years = st.number_input("Loan Term (Years)", min_value=1, max_value=30)

    with col2:
        credit_history = st.radio("Credit History", ["Yes", "No"])
        employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed"])
        property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

    st.markdown("<div class='center-btn'>", unsafe_allow_html=True)
    submit = st.form_submit_button("✅ Check Loan Eligibility")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Model Architecture Card
# -----------------------------
st.markdown("<h2>🧩 Model Architecture (Stacking)</h2>", unsafe_allow_html=True)

st.markdown("""
<ul>
<li><b>Base Models</b>
    <ul>
        <li>Logistic Regression</li>
        <li>Decision Tree</li>
        <li>Random Forest</li>
    </ul>
</li>
<li><b>Meta Model</b>: Logistic Regression</li>
</ul>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if submit:
    credit_val = 1 if credit_history == "Yes" else 0
    emp_val = 1 if employment == "Self-Employed" else 0

    property_map = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}
    property_val = property_map[property_area]

    # Convert years → months (matches training distribution)
    loan_term = loan_term_years * 12

    input_data = np.array([[
        1, 1, 0, 1,
        emp_val,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_val,
        property_val
    ]])

    # scale numeric features
    input_data[:, [5, 6, 7]] = scaler.transform(input_data[:, [5, 6, 7]])

    proba = stack_model.predict_proba(input_data)[0]
    confidence = proba[1] * 100   # probability of approval

    prediction = 1 if proba[1] >= 0.45 else 0


    proba = stack_model.predict_proba(input_data)[0]
    confidence = max(proba) * 100

    # -----------------------------
    # Result Card
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>📊 Prediction Result</h2>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("<div class='approved'>✅ Loan Approved</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='rejected'>❌ Loan Rejected</div>", unsafe_allow_html=True)

    st.markdown(
        f"<p class='confidence'>📈 Confidence Score: <b>{confidence:.2f}%</b></p>",
        unsafe_allow_html=True
    )

    # -----------------------------
    # Business Explanation
    # -----------------------------
    st.markdown("<h3>🧠 Business Explanation</h3>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            "<p class='explain'>"
            "Based on income, credit history, loan term, and combined predictions "
            "from multiple models, the applicant is likely to repay the loan. "
            "Hence, the system recommends <b>loan approval</b>."
            "</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p class='explain'>"
            "Based on income, credit history, loan term, and combined predictions "
            "from multiple models, the applicant is unlikely to repay the loan. "
            "Hence, the system recommends <b>loan rejection</b>."
            "</p>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
