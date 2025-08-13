import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Grocery Data Analysis & Prediction", layout="wide")
st.title("Grocery Data Analysis & Prediction App")
st.write("Explore data, visualize insights, and make predictions using a trained ML model.")

# -----------------------------
# Sidebar Navigation
# -----------------------------
section = st.sidebar.radio(
    "Select Section",
    ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"]
)

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = "data/cleaned_dataset.csv"
MODEL_PATH = "models/best_model.joblib"

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at `{DATA_PATH}`. Please upload it.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Load Trained Model
# -----------------------------
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.warning(f"Model not found at `{MODEL_PATH}`. Predictions will be disabled.")
    model = None

# -----------------------------
# Data Exploration Section
# -----------------------------
if section == "Data Exploration":
    st.header("Dataset Overview")
    st.subheader("Shape & Columns")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.write(df.head())

    st.subheader("Filter Data")
    col_to_filter = st.selectbox("Select Column to Filter", df.columns)
    unique_vals = df[col_to_filter].dropna().unique()
    selected_vals = st.multiselect("Select values", unique_vals, default=unique_vals[:5])
    st.write(df[df[col_to_filter].isin(selected_vals)])

# -----------------------------
# Visualizations Section
# -----------------------------
elif section == "Visualizations":
    st.header("Data Visualizations")

    st.subheader("Bar Plot")
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        col1 = st.selectbox("Select Column for Bar Plot", cat_cols)
        bar_data = df[col1].value_counts().reset_index()
        bar_data.columns = [col1, "count"]
        fig1 = px.bar(bar_data, x=col1, y="count", title=f"Bar Plot of {col1}")
        st.plotly_chart(fig1)
    else:
        st.info("No categorical columns available for bar plot.")

    st.subheader("Histogram")
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        num_col = st.selectbox("Select Numeric Column for Histogram", num_cols)
        fig2 = px.histogram(df, x=num_col, nbins=20, title=f"Histogram of {num_col}")
        st.plotly_chart(fig2)
    else:
        st.info("No numeric columns available for histogram.")

    st.subheader("Correlation Heatmap")
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig3, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig3)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

# -----------------------------
# Model Prediction Section
# -----------------------------
elif section == "Model Prediction":
    st.header("Predict Outcome")

    if model is None:
        st.warning("Model not loaded. Cannot make predictions.")
    else:
        # Auto-detect features (excluding target)
        feature_cols = [col for col in df.columns if col.lower() not in ["target", "label", "high_spend"]]
        input_data = {}
        for col in feature_cols:
            if df[col].dtype == 'object':
                input_data[col] = st.selectbox(f"Select {col}", df[col].dropna().unique())
            else:
                input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):
            prediction = model.predict(input_df)
            st.success(f"Predicted outcome: {prediction[0]}")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)
                st.write("Prediction probabilities:")
                st.write(proba)

# -----------------------------
# Model Performance Section
# -----------------------------
elif section == "Model Performance":
    st.header("Model Performance")

    if model is None:
        st.warning("Model not loaded. Cannot show performance.")
    else:
        st.subheader("Evaluation Metrics")
        st.write("Add evaluation metrics here after testing model on a holdout set.")

        st.subheader("Confusion Matrix / Charts")
        st.write("Plot confusion matrix or ROC curves here.")

        st.subheader("Model Comparison")
        st.write("If you trained multiple models, compare them here.")
