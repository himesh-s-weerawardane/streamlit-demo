import streamlit as st
import pickle
import numpy as np

# 1. Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Page title
st.title("Customer Purchase Prediction Demo")
st.write("Enter customer details to predict whether they will purchase.")

# 3. Input widgets
age = st.slider("Age", 18, 80, 30)
income = st.slider("Annual Income (thousands)", 10, 200, 50)
past_spend = st.slider("Past Spend Amount", 0, 1000, 100)

# 4. Prediction button
if st.button("Predict"):
    features = np.array([[age, income, past_spend]])
    prob = model.predict_proba(features)[0][1]
    pred = model.predict(features)[0]

    st.subheader("Prediction Result")
    st.write(f"Purchase Probability: **{prob:.2%}**")
    if pred == 1:
        st.success("Prediction: This customer is likely to purchase ✅")
    else:
        st.warning("Prediction: This customer is unlikely to purchase ❌")