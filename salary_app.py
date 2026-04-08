# ==============================
# Imports
# ==============================
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Salary Prediction", layout="wide")

# ==============================
# Background Styling 
# ==============================
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
        color: black;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Load Model, Encoders, Features
# ==============================
model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")
features = joblib.load("features.pkl")

# ==============================
# Load Dataset (for dropdowns)
# ==============================
df = pd.read_csv("Jobs_NYC_Postings.csv")  

# ==============================
# Column Cleaning 
# ==============================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r'[^a-z0-9]', '_', regex=True)
    .str.replace(r'_+', '_', regex=True)
    .str.strip('_')
)

# ==============================
# Feature Engineering
# ==============================
if 'posting_date' in df.columns:
    df['posting_date'] = pd.to_datetime(df['posting_date'])
    df['posting_year'] = df['posting_date'].dt.year
    df['posting_month'] = df['posting_date'].dt.month

# ==============================
# Title
# ==============================
st.title("💼 Salary Prediction App")
st.write("Enter job details to predict salary")

# ==============================
# Side-by-Side Inputs 
# ==============================
col1, col2 = st.columns(2)

with col1:
    agency = st.selectbox("Agency", sorted(df['agency'].dropna().unique()))
    posting_type = st.selectbox("Posting Type", sorted(df['posting_type'].dropna().unique()))
    of_positions = st.number_input("Number of Positions", min_value=1, value=1)
    level = st.selectbox("Level", sorted(df['level'].dropna().unique()))
    job_category = st.selectbox("Job Category", sorted(df['job_category'].dropna().unique()))
    full_time = st.selectbox("Full-Time / Part-Time", sorted(df['full_time_part_time_indicator'].dropna().unique()))

with col2:
    career_level = st.selectbox("Career Level", sorted(df['career_level'].dropna().unique()))
    salary_frequency = st.selectbox("Salary Frequency", sorted(df['salary_frequency'].dropna().unique()))
    residency_requirement = st.selectbox("Residency Requirement", sorted(df['residency_requirement'].dropna().unique()))
    posting_year = st.selectbox("Posting Year", sorted(df['posting_year'].dropna().unique()))
    posting_month = st.slider("Posting Month", 1, 12, 6)
    work_location = st.selectbox("Work Location", sorted(df['work_location'].dropna().unique()))
    division_work_unit = st.selectbox("Division Work Unit", sorted(df['division_work_unit'].dropna().unique()))

# ==============================
# Create Input DataFrame
# ==============================
input_data = pd.DataFrame({
    'agency': [agency],
    'posting_type': [posting_type],
    'of_positions': [of_positions],
    'level': [level],
    'job_category': [job_category],
    'full_time_part_time_indicator': [full_time],
    'career_level': [career_level],
    'salary_frequency': [salary_frequency],
    'work_location': [work_location],
    'division_work_unit': [division_work_unit],
    'residency_requirement': [residency_requirement],
    'posting_year': [posting_year],
    'posting_month': [posting_month]
})

# ==============================
# Encoding
# ==============================
for col in input_data.columns:
    if col in encoders:
        try:
            input_data[col] = encoders[col].transform(input_data[col])
        except:
            st.error(f"Invalid value for {col}")
            st.stop()

# ==============================
# Feature Alignment 
# ==============================
input_data = input_data.reindex(columns=features, fill_value=0)

# ==============================
# Prediction
# ==============================
if st.button("🚀 Predict Salary"):
    
    prediction = model.predict(input_data)[0]
    
    prediction = np.expm1(prediction)
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(to right, #000428, #004e92);
            padding:20px;
            border-radius:12px;
            text-align:center;
            font-size:26px;
            color:white;
            font-weight:bold;">
            💰 Predicted Salary: ${prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )