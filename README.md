# Multiple_Linear_Regression_App
An interactive Streamlit web application for performing Multiple Linear Regression (MLR) on any numeric dataset.
The app allows users to upload a CSV file, select multiple feature columns and a target column, train a regression model, visualize relationships, and make predictions - all in one simple interface.


#  🔥Live Demo

👉 Launch the App- https://multiplelinearregressionapp.streamlit.app/

(Runs directly on Streamlit Cloud - no setup needed!)

#  Features

- Upload any numeric CSV dataset
- Select input (X) and output (Y) columns interactively
- Automatically train a Multiple Linear Regression model

 # Visual Insights:
- Correlation heatmap among selected columns
- Custom Prediction – enter feature values to get predicted output
- No coding required! Just upload -> select -> analyze.

 # Example Use Cases
- Salary prediction based on experience, education, and skill ratings
- House price prediction using area, rooms, and location features
- Medical or scientific datasets with multiple numeric predictors
- Any regression problem with one target variable and multiple numeric features

 # How to run locally (Run below code line step by step)
  - Clone the repository
    - git clone https://github.com/hemant-pm/Multiple_Linear_Regression_App.git

  - Navigate to project directory
    - cd Multiple_Linear_Regression_App

  - Install required dependencies
    - pip install -r requirements.txt

  - Run the Streamlit app
    - streamlit run MLR_Project/MLR.py
   
  #  Folder Structure
- Multiple_Linear_Regression_App/
  - MLR_Project
    - MLR.py               # Main Streamlit app
  - requirements.txt         # Dependencies list
  - README.md                # Project documentation

#  Workflow Overview

- Upload Dataset (.csv)
- Select Features (X) and Target (Y)
- Train Linear Regression Model
- View Correlation Plots
- Predict Output for New Inputs

#  🎉 Learning Outcome
- Understand and apply Multiple Linear Regression
- Visualize data correlations
- Deploy machine learning apps using Streamlit Cloud
- Perform custom predictions using trained models
