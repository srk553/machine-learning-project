import streamlit as st
import pickle
import numpy as np

# Set page title and layout
st.set_page_config(page_title="SRK Masterstack - Salary Predictor", page_icon="💰", layout="centered")

# Custom styling
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2E86C1;
        }
        .subtext {
            text-align: center;
            font-size: 18px;
            color: #5D6D7E;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>SRK Masterstack</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Based on experience, education, and country, the application makes salary predictions.</p>", unsafe_allow_html=True)

st.divider()

# Input options
country = [
    "United States of America", "Germany", "United Kingdom of Great Britain and Northern Ireland", 
    "Ukraine", "India", "France", "Canada", "Brazil", "Spain", "Italy", "Netherlands", "Australia", "Other"
]

education = [
    "Bachelor’s degree", "Master’s degree", "Post Grad", "Less than a Degree"
]

col1, col2 = st.columns(2)
with col1:
    selected_country = st.selectbox("🌍 Select Your Country", country)
with col2:
    selected_education = st.selectbox("🎓 Select Your Education Level", education)

selected_exp = st.slider("💼 Experience in Years", 1, 50, value=5)

st.divider()

# Load model
with open('saved_model.pk1', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Process input
X = np.array([[selected_country, selected_education, selected_exp]])
X[:, 0] = le_country.transform(X[:, 0])
X[:, 1] = le_education.transform(X[:, 1])
X = X.astype(float)

# Predict salary
y_pred = regressor_loaded.predict(X)

# Display result
st.success("✨ Prediction Complete!")
st.subheader("💰 The predicted salary is: **${:,.0f}**".format(y_pred[0]))

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: grey;'>Developed by <strong>SRK Masterstack</strong></p>
""", unsafe_allow_html=True)