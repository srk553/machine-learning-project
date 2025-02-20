import streamlit as st

st.title("SRK Masterstack")

st.info("Based on experience, education, and country, the application makes predictions.")

country = {
    "United States of America",
    "Germany",
    "United Kingdom of Great Britain and Northern Ireland",
    "Ukraine",
    "India",
    "France",
    "Canada",
    "Brazil",
    "Spain",
    "Italy",
    "Netherlands",
    "Australia",
    "Other"
}

education= {
    "Bachelor’s degree",
    "Master’s degree",
    "Post Grad",
    "Less than a Degree"
}

selected_country = st.selectbox("Country",country)
selected_education = st.selectbox("Education",education)
selected_exp = st.slider("Experience in Years",1,50)

import pickle
import numpy as np

with open('saved_model.pk1', 'rb') as file:
  data = pickle.load(file)

regressor_loaded =  data["model"]
le_country =  data["le_country"]
le_education =  data["le_education"]

X = np.array([[selected_country,selected_education,selected_exp]])
X[:,0] = le_country.transform(X[:,0])
X[:,1] = le_education.transform(X[:,1])
X = X.astype(float)
# print(X)

y_pred = regressor_loaded.predict(X)
# print(y_pred)

st.subheader("The predicted Salary is $ {0:,.0f}".format(y_pred[0]))