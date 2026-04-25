
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# Set page config
st.set_page_config(page_title="Tourism Purchase Predictor", layout="centered")

# Repository details for the model on Hugging Face Hub
MODEL_REPO_ID = "Lalithas/Tourism-Prediction-Model"
MODEL_FILENAME = "best_random_forest_model.joblib"

# Create a directory to store the downloaded model if it doesn't exist
model_dir = "./model"
os.makedirs(model_dir, exist_ok=True)

# Path to store the downloaded model
local_model_path = os.path.join(model_dir, MODEL_FILENAME)

# Function to load the model
@st.cache_resource
def load_model():
    try:
        # Download the model from Hugging Face Hub
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=model_dir,
            local_dir_use_symlinks=False # Set to False for colab environment
        )
        model = joblib.load(local_model_path)
        st.success("Model loaded successfully from Hugging Face Hub!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# App title and description
st.title("🌍 Tourism Package Purchase Predictor")
st.markdown("### Predict if a customer will purchase the Wellness Tourism Package")

st.write("""
    This application predicts the likelihood of a customer purchasing the Wellness Tourism Package.
    Please provide the customer details below.
""")

# Input fields for customer details
st.header("Customer Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 30)
        typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"], index=0)
        citytier = st.selectbox("City Tier", [1, 2, 3], index=0)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"], index=0)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        numberofpersonvisiting = st.slider("Number of Persons Visiting", 1, 10, 2)
        preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5], index=0)
    with col2:
        maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0)
        numberoftrips = st.slider("Number of Trips Annually", 0, 20, 2)
        passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=1)
        owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        numberofchildrenvisiting = st.slider("Number of Children Visiting (Age < 5)", 0, 5, 0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "CEO"], index=0)
        monthlyincome = st.number_input("Monthly Income", min_value=10000, max_value=1000000, value=50000)

    st.subheader("Customer Interaction Data")
    col3, col4 = st.columns(2)
    with col3:
        pitchsatisfactionsore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"], index=2)
    with col4:
        numberoffollowups = st.slider("Number of Follow-ups", 0, 10, 2)
        durationofpitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)

    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [typeofcontact],
            'CityTier': [citytier],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [numberofpersonvisiting],
            'PreferredPropertyStar': [preferredpropertystar],
            'MaritalStatus': [maritalstatus],
            'NumberOfTrips': [numberoftrips],
            'Passport': [passport],
            'OwnCar': [owncar],
            'NumberOfChildrenVisiting': [numberofchildrenvisiting],
            'Designation': [designation],
            'MonthlyIncome': [monthlyincome],
            'PitchSatisfactionScore': [pitchsatisfactionsore],
            'ProductPitched': [productpitched],
            'NumberOfFollowups': [numberoffollowups],
            'DurationOfPitch': [durationofpitch]
        })

        # Encode categorical features - MUST match the preprocessing in data_preparation.py
        # The order and mapping of categories must be consistent
        # In data_preparation.py, these were encoded using LabelEncoder
        # 'TypeofContact': Company Invited -> 0, Self Inquiry -> 1, Unknown -> 2
        # 'Gender': Female -> 0, Male -> 1, Unknown -> 2
        # 'ProductPitched': Basic -> 0, Deluxe -> 1, King -> 2, Standard -> 3, Super Deluxe -> 4
        # 'MaritalStatus': Divorced -> 0, Married -> 1, Single -> 2
        # 'Designation': AVP -> 0, CEO -> 1, Executive -> 2, Manager -> 3, Senior Manager -> 4, VP -> 5
        # 'Occupation': Freelancer -> 0, Large Business -> 1, Salaried -> 2, Small Business -> 3, Unknown -> 4

        type_of_contact_mapping = {"Company Invited": 0, "Self Inquiry": 1, "Unknown": 2}
        gender_mapping = {"Female": 0, "Male": 1, "Unknown": 2}
        product_pitched_mapping = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
        marital_status_mapping = {"Divorced": 0, "Married": 1, "Single": 2}
        designation_mapping = {"AVP": 0, "CEO": 1, "Executive": 2, "Manager": 3, "Senior Manager": 4, "VP": 5}
        occupation_mapping = {"Freelancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3, "Unknown": 4}

        input_data['TypeofContact'] = input_data['TypeofContact'].map(type_of_contact_mapping)
        input_data['Gender'] = input_data['Gender'].map(gender_mapping)
        input_data['ProductPitched'] = input_data['ProductPitched'].map(product_pitched_mapping)
        input_data['MaritalStatus'] = input_data['MaritalStatus'].map(marital_status_mapping)
        input_data['Designation'] = input_data['Designation'].map(designation_mapping)
        input_data['Occupation'] = input_data['Occupation'].map(occupation_mapping)

        # CRITICAL: Force the feature order to match model.feature_names_in_
        if hasattr(model, "feature_names_in_"):
            input_data = input_data[model.feature_names_in_]
        
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[:, 1][0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success(f"The customer is likely to purchase the package! (Probability: {prediction_proba:.2f})")
        else:
            st.info(f"The customer is not likely to purchase the package. (Probability: {prediction_proba:.2f})")

        st.write("""
            **Note**: This is a predictive model's output and should be used as guidance.
            Further analysis and business context are always recommended.
        """)

