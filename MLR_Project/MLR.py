
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="📊 Multiple Linear Regression App", layout="wide")

st.title("💼 Multiple Linear Regression App (Any Dataset!)")

# ======= SIDEBAR INPUT =======
st.sidebar.header("🚀 User Options")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    # Show dataset in main area
    st.write("### 📊 Data Preview")
    st.write(df.head())
    st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

    # ======= FEATURE SELECTION =======
    st.sidebar.subheader("Select Features and Target")
    all_columns = df.columns.tolist()

    # Multi-select for features
    feature_columns = st.sidebar.multiselect("Select Feature Columns (X)", options=all_columns, default=all_columns[:-1])
    target_column = st.sidebar.selectbox("Select Target Column (y)", options=all_columns, index=len(all_columns)-1)

    if len(feature_columns) < 1:
        st.warning("⚠ Please select at least one feature column!")
    else:
        X = df[feature_columns]
        y = df[target_column]

        # ======= TRAIN-TEST SPLIT =======
        test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # ======= MODEL TRAINING =======
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ======= MODEL EVALUATION =======
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        st.write("### 📊 Model Performance")
        st.write(f"**Mean Squared Error (MSE):** {mse:,.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:,.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # ======= USER INPUT PREDICTION =======
        st.sidebar.subheader("Predict New Data")
        user_input = {}
        for col in feature_columns:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            step_val = (max_val - min_val)/100 if max_val != min_val else 1.0
            user_input[col] = st.sidebar.number_input(f"{col}:", min_value=min_val, max_value=max_val, step=step_val, value=min_val)

        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        st.sidebar.success(f"Predicted {target_column}: {prediction[0]:,.2f}")

        # ======= VISUALIZATIONS =======
        st.write("### 📉 Visualizations")
        fig_size = (5, 4)

        # Correlation heatmap
        st.write("#### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(df[feature_columns + [target_column]].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Histogram of target
        st.write("#### 📊 Histogram of Target")
        fig, ax = plt.subplots(figsize=fig_size)
        sns.histplot(y, kde=True, ax=ax)
        st.pyplot(fig)

else:
    st.warning("⚠ Please upload a CSV file to proceed.")

