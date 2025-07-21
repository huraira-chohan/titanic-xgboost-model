# app.py

import streamlit as st
import pandas as pd
import joblib
import os # Import os to check for file existence

# --- Model Loading ---
# Define the path to the trained model file
MODEL_PATH = "xgb_titanic_model.pkl"

# Load the trained model
try:
    # Check if the model file exists before attempting to load
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: The model file '{MODEL_PATH}' was not found.")
        st.error("Please ensure the model file is in the same directory as 'app.py'.")
        st.stop() # Stop the app execution if the model is not found

    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.error("Please ensure 'xgb_titanic_model.pkl' is a valid joblib model file.")
    st.stop() # Stop the app execution if model loading fails

# --- App Title and Introduction ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")
st.markdown("""
    Predict survival based on Titanic passenger details using a trained XGBoost model.
    Enter the passenger's information below and click 'Predict' to see the outcome.
""")

# --- Input Fields ---
st.header("📝 Enter Passenger Details")

# Passenger Class
Pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3],
    help="1 = 1st Class (Upper), 2 = 2nd Class (Middle), 3 = 3rd Class (Lower)"
)

# Sex
Sex = st.radio(
    "Sex",
    ['male', 'female']
)

# Age
Age = st.slider(
    "Age",
    min_value=0,
    max_value=80,
    value=25,
    step=1,
    help="Passenger's age in years."
)

# Family Members (SibSp + Parch)
Family = st.slider(
    "Number of Family Members (SibSp + Parch)",
    min_value=0,
    max_value=10,
    value=1,
    step=1,
    help="Total number of siblings/spouses (SibSp) and parents/children (Parch) aboard."
)

# Fare Paid
Fare = st.slider(
    "Fare Paid ($)",
    min_value=0.0,
    max_value=600.0,
    value=30.0,
    step=0.5,
    help="The fare paid for the ticket."
)

# Port of Embarkation
Embarked = st.selectbox(
    "Port of Embarkation",
    ['C', 'Q', 'S'],
    help="C = Cherbourg, Q = Queenstown, S = Southampton"
)

# --- Prediction Function ---
def predict_survival(Pclass, Sex, Age, Family, Fare, Embarked):
    # Create a dictionary with input data, matching the structure for one-hot encoding
    # and feature names used during training.
    # Note: 'Sex_male', 'Embarked_Q', 'Embarked_S' are created here as per training script.
    input_data = {
        'Pclass': Pclass,
        'Age': Age,
        'Family': Family, # This is the combined SibSp + Parch
        'Fare': Fare,
        'Sex_male': 1 if Sex == 'male' else 0,
        'Embarked_Q': 1 if Embarked == 'Q' else 0,
        'Embarked_S': 1 if Embarked == 'S' else 0,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # IMPORTANT: Ensure the columns are in the same order as the training data (X_train)
    # The original training script's `predict_custom` function used `df_input[X_train.columns]`.
    # To replicate this, we need the exact column names and their order from the training.
    # Based on the training script's `X = df_encoded.iloc[:,1:]`, the columns are:
    # 'Pclass', 'Age', 'Fare', 'Family', 'Sex_male', 'Embarked_Q', 'Embarked_S'
    # We explicitly define this order to ensure consistency.
    expected_columns = ['Pclass', 'Age', 'Fare', 'Family', 'Sex_male', 'Embarked_Q', 'Embarked_S']

    # Reindex the input DataFrame to match the training data's column order
    # Any missing columns (e.g., if we had more Embarked categories and drop_first=False)
    # would be filled with NaN by default, which is then handled by XGBoost.
    # However, since we explicitly create all expected one-hot encoded columns,
    # this step primarily ensures correct ordering.
    input_df = input_df[expected_columns]


    # Predict using the loaded model
    prediction = model.predict(input_df)[0]

    # Return a user-friendly result
    if prediction == 1:
        return "🟢 Survived"
    else:
        return "🔴 Did NOT Survive"

# --- Trigger Prediction ---
st.markdown("---") # Separator for better UI
if st.button("Predict Survival", help="Click to get the prediction based on the entered details."):
    with st.spinner('Predicting...'):
        result = predict_survival(Pclass, Sex, Age, Family, Fare, Embarked)
        st.subheader("Prediction Result:")
        if "Survived" in result:
            st.success(result)
            st.balloons() # Add a little celebration for survival
        else:
            st.error(result)

st.markdown("---")
st.caption("This predictor is for demonstration purposes based on a trained machine learning model.")
