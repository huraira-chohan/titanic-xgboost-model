# app.py

import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load("xgb_titanic_model.pkl")
except FileNotFoundError:
    st.error("Error: The model file 'xgb_titanic_model.pkl' was not found.")
    st.error("Please ensure the model file is in the same directory as 'app.py'.")
# Load the trained model
model = joblib.load("xgb_titanic_model.pkl")

# App title and intro
st.title("🚢 Titanic Survival Predictor")
st.markdown("Predict survival based on Titanic passenger details using a trained XGBoost model.")

# Input fields
st.header("📝 Enter Passenger Details")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.radio("Sex", ['male', 'female'])
Age = st.slider("Age", 0, 80, 25)
Family = st.slider("Number of Family Members (SibSp + Parch)", 0, 10, 1)
Fare = st.slider("Fare Paid ($)", 0.0, 600.0, 30.0)
Embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Prediction function
def predict_survival(Pclass, Sex, Age, Family, Fare, Embarked):
    input_df = pd.DataFrame([{
        'Pclass': Pclass,
        'Age': Age,
        'Family': Family,
        'Fare': Fare,
        'Sex_male': 1 if Sex == 'male' else 0,
        'Embarked_Q': 1 if Embarked == 'Q' else 0,
        'Embarked_S': 1 if Embarked == 'S' else 0
    }])

    # Predict using loaded model
    prediction = model.predict(input_df)[0]
    return "🟢 Survived" if prediction == 1 else "🔴 Did NOT Survive"

# Trigger prediction
if st.button("Predict"):
    result = predict_survival(Pclass, Sex, Age, Family, Fare, Embarked)
    st.subheader("Prediction:")
    st.success(result)
