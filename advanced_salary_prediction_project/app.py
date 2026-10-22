from pathlib import Path
from datetime import datetime
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

DATA_FILE = Path("salary_data.csv")
MODEL_FILE = Path("salary_model.pkl")
HISTORY_FILE = Path("prediction_history.csv")
COMPARISON_FILE = Path("model_comparison.csv")

FEATURES = ["Experience", "Education", "Age", "Skills", "JobRole"]
TARGET = "Salary"
JOB_ROLES = ["Software Engineer", "Data Analyst", "Embedded Engineer"]

st.set_page_config(page_title="Salary Prediction System", page_icon="💼", layout="wide")
st.title("Web-Based Salary Prediction System")
st.caption("Machine Learning project using Random Forest Regression")
st.info(
    "Educational demo: estimates depend on the dataset used for training. "
    "They are not guaranteed job offers or verified market salary figures."
)

if not MODEL_FILE.exists() or not COMPARISON_FILE.exists():
    st.error("Trained model files are missing. Run `python train_model.py` in the project folder first.")
    st.stop()

model = joblib.load(MODEL_FILE)
df = pd.read_csv(DATA_FILE)
if "JobRole" not in df.columns:
    df["JobRole"] = "Software Engineer"

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Predict Salary", "Compare ML Models", "Prediction History", "Sample Dataset"]
)

def create_pdf(name, experience, education, age, skills, role, salary):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 70
    pdf.setTitle("Salary Prediction Report")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(55, y, "Salary Prediction Report")
    y -= 35
    pdf.setFont("Helvetica", 11)
    pdf.drawString(55, y, f"Report date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 35
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(55, y, "Input Details")
    y -= 22
    pdf.setFont("Helvetica", 11)
    details = [
        f"Name: {name or 'Not provided'}",
        f"Experience: {experience} years",
        f"Education: {education}",
        f"Age: {age}",
        f"Main skill: {skills}",
        f"Job role: {role}",
        f"Estimated annual salary: INR {salary:,.0f}",
    ]
    for item in details:
        pdf.drawString(60, y, item)
        y -= 22
    y -= 15
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(55, y, "Educational estimate only; not a verified market salary or job offer.")
    pdf.drawString(55, y - 15, "Prediction quality depends on the dataset and model evaluation.")
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

if page == "Predict Salary":
    st.subheader("Enter Your Details")
    with st.form("salary_form"):
        name = st.text_input("Name (optional)")
        col1, col2 = st.columns(2)
        with col1:
            experience = st.number_input("Experience (years)", min_value=0, max_value=50, value=2, step=1)
            education = st.selectbox("Education Level", ["UG", "PG", "Diploma", "Other"])
            age = st.number_input("Age", min_value=18, max_value=70, value=23, step=1)
        with col2:
            skills = st.selectbox("Main Skill", ["Python", "C++", "Java", "Data Science", "VLSI", "IoT", "Embedded C"])
            role = st.selectbox("Job Role", JOB_ROLES)
        submitted = st.form_submit_button("Predict Salary", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "Experience": experience,
            "Education": education,
            "Age": age,
            "Skills": skills,
            "JobRole": role
        }], columns=FEATURES)
        salary = max(0, float(model.predict(row)[0]))
        st.success(f"Estimated Annual Salary: ₹{salary:,.0f}")
        st.caption("This is a model estimate, not a guaranteed salary offer.")

        history_row = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name": name,
            "Experience": experience,
            "Education": education,
            "Age": age,
            "Skills": skills,
            "JobRole": role,
            "EstimatedSalary": round(salary, 2)
        }
        new_history = pd.DataFrame([history_row])
        if HISTORY_FILE.exists():
            old_history = pd.read_csv(HISTORY_FILE)
            history = pd.concat([old_history, new_history], ignore_index=True)
        else:
            history = new_history
        history.to_csv(HISTORY_FILE, index=False)

        pdf_bytes = create_pdf(name, experience, education, age, skills, role, salary)
        st.download_button(
            "Download Salary Report (PDF)",
            data=pdf_bytes,
            file_name="salary_prediction_report.pdf",
            mime="application/pdf"
        )
        st.toast("Prediction saved to history.")

elif page == "Compare ML Models":
    st.subheader("Compare Machine Learning Algorithms")
    comparison = pd.read_csv(COMPARISON_FILE)
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption("Lower MAE/RMSE is generally better. R² closer to 1 is generally better. Tiny/demo datasets can make these scores unstable.")
    metric = st.selectbox("Choose metric to plot", ["RMSE (₹)", "MAE (₹)", "R² Score"])
    fig, ax = plt.subplots()
    ax.bar(comparison["Model"], comparison[metric])
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison: {metric}")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Prediction History":
    st.subheader("Prediction History")
    if HISTORY_FILE.exists():
        history = pd.read_csv(HISTORY_FILE)
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            "Download History (CSV)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
        if st.button("Clear Prediction History"):
            HISTORY_FILE.unlink(missing_ok=True)
            st.success("History file deleted. Refresh the page to see the empty history.")
    else:
        st.info("No predictions saved yet. Open Predict Salary and make a prediction first.")

elif page == "Sample Dataset":
    st.subheader("Sample Dataset")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="salary_data.csv",
        mime="text/csv"
    )
    st.caption("Check the dataset source and quality before using predictions for real decisions.")
