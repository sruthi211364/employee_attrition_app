import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the dataset
data = pd.read_csv("data_final.csv", index_col=0)

# Drop unnecessary columns
columns_to_drop = ["IncomePerYearWorked", "SatisfactionIndex"]
data = data.drop(columns=columns_to_drop, errors='ignore')

# Split features and target
x_final = data.drop(columns=["Attrition"], errors='ignore')
y_final = data["Attrition"]

# Identify numerical and categorical features
numerical_features = x_final.select_dtypes(include=["int64", "float64"]).columns

# Safe log transformation
def safe_log_transform(X):
    return np.log1p(np.maximum(X, 0))

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("minmax", MinMaxScaler()),
            ("log", FunctionTransformer(safe_log_transform, validate=True))
        ]), numerical_features)
    ]
)

# Classifier pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_jobs=10, n_neighbors=5, metric='minkowski'))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(pipeline, "attrition_model.joblib")

# FastAPI application
from fastapi import FastAPI

app = FastAPI()

# Load the model
model = joblib.load("attrition_model.joblib")

@app.post("/predict")
async def predict(input_data: dict):
    """Endpoint to make predictions."""
    input_df = pd.DataFrame([input_data])

    # Convert categorical fields to numeric manually
    input_df["Gender"] = input_df["Gender"].map({"Male": 0, "Female": 1})
    input_df["EducationField"] = input_df["EducationField"].map({"Life Sciences": 0, "Medical": 1, "Marketing": 2, "Technical Degree": 3, "Other": 4})
    input_df["MaritalStatus"] = input_df["MaritalStatus"].map({"Single": 0, "Married": 1, "Divorced": 2})
    input_df["JobRole"] = input_df["JobRole"].map({"Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2, "Manager": 3, "Sales Representative": 4, "Research Director": 5, "Human Resources": 6, "Healthcare Representative": 7, "Manufacturing Director": 8})
    input_df["Department"] = input_df["Department"].map({"Sales": 0, "Research & Development": 1, "Human Resources": 2})

    # Validate input data
    input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    processed_input = model.named_steps['preprocessor'].transform(input_df)
    prediction = model.named_steps['classifier'].predict(processed_input)
    return {"prediction": int(prediction[0])}

##########################################
# Streamlit App
import streamlit as st

st.title("Employee Attrition Prediction")

st.write("Provide the following details to predict employee attrition:")

# Input fields
input_data = {
    "Age": st.number_input("Age", min_value=18, max_value=70, step=1),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "DistanceFromHome": st.number_input("Distance From Home", min_value=0, max_value=100, step=1),
    "EducationField": st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"]),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
    "JobRole": st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager", "Sales Representative", "Research Director", "Human Resources", "Healthcare Representative", "Manufacturing Director"]),
    "Department": st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"]),
    "JobLevel": st.number_input("Job Level", min_value=1, max_value=5, step=1),
    "EnvironmentSatisfaction": st.number_input("Environment Satisfaction", min_value=1, max_value=4, step=1),
    "JobSatisfaction": st.number_input("Job Satisfaction", min_value=1, max_value=4, step=1),
    "WorkLifeBalance": st.number_input("Work Life Balance", min_value=1, max_value=4, step=1),
    "StockOptionLevel": st.number_input("Stock Option Level", min_value=0, max_value=3, step=1),
    "TotalWorkingYears": st.number_input("Total Working Years", min_value=0, max_value=50, step=1),
    "YearsInCurrentRole": st.number_input("Years in Current Role", min_value=0, max_value=30, step=1),
    "NumCompaniesWorked": st.number_input("Number of Companies Worked", min_value=0, max_value=15, step=1),
    "PercentSalaryHike": st.number_input("Percent Salary Hike", min_value=0, max_value=100, step=1),
    "DailyRate": st.number_input("Daily Rate", min_value=0, max_value=2000, step=1),
    "MonthlyIncome": st.number_input("Monthly Income", min_value=0, max_value=50000, step=100),
}

if st.button("Predict Attrition"):
    input_df = pd.DataFrame([input_data])

    # Convert categorical fields to numeric manually
    input_df["Gender"] = input_df["Gender"].map({"Male": 0, "Female": 1})
    input_df["EducationField"] = input_df["EducationField"].map({"Life Sciences": 0, "Medical": 1, "Marketing": 2, "Technical Degree": 3, "Other": 4})
    input_df["MaritalStatus"] = input_df["MaritalStatus"].map({"Single": 0, "Married": 1, "Divorced": 2})
    input_df["JobRole"] = input_df["JobRole"].map({"Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2, "Manager": 3, "Sales Representative": 4, "Research Director": 5, "Human Resources": 6, "Healthcare Representative": 7, "Manufacturing Director": 8})
    input_df["Department"] = input_df["Department"].map({"Sales": 0, "Research & Development": 1, "Human Resources": 2})

    # Validate input data
    input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    processed_input = model.named_steps['preprocessor'].transform(input_df)
    prediction = model.named_steps['classifier'].predict(processed_input)
    st.write("Prediction: ", "Employee is not willing to leave the company." if prediction[0] == 1 else "Employee is likely to leave.")
