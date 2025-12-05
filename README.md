# Employee Attrition Prediction – Machine Learning Project

This project predicts whether an employee is likely to leave the organization using Machine Learning models. It includes complete data preprocessing, EDA, feature engineering, model training, evaluation, and a Streamlit web application for real-time prediction.

**Project Structure**

├── emp_attrition.py          # Final Streamlit application + model training

├── emp_attrition.csv         # Cleaned dataset

├── le.pkl                    # Saved label encoders

├── ohe.pkl                   # Saved One Hot Encoder

├── scaler.pkl                # StandardScaler

├── lr.pkl                    # Logistic Regression model

├── knn.pkl                   # K-Nearest Neighbors

├── rf.pkl                    # Random Forest model

├── dt.pkl                    # Decision Tree model

├── gb.pkl                    # Gradient Boosting model

├── README.md                 # Project documentation

**Project Overview**

Employee attrition creates financial loss for companies in terms of recruitment, training, and productivity.
This project helps HR teams:

✔ Identify employees at risk
✔ Understand attrition-related factors
✔ Take preventive actions

**1. Data Preprocessing**

Steps performed:

Removed unwanted columns like Unnamed: 0

Handled categorical variables using Label Encoding

Numerical scaling using StandardScaler

Split dataset into Train–Test using 80/20 ratio

Saved preprocessors (encoders.pkl, scaler.pkl) for use in Streamlit app

**2. Exploratory Data Analysis (EDA)**

EDA included:

📌 Univariate Analysis

Distribution of Age, Income, JobSatisfaction, etc.

Attrition count plot

📌 Bivariate / Relationship Plots

Attrition vs Age

Attrition vs Monthly Income

Attrition vs Total Working Years

Attrition vs OverTime

Attrition vs Job Role

Correlation heatmap,etc...

These help understand which features strongly influence attrition.

**3. Feature Engineering**

Encoded categorical features

Scaled numerical features

Removed unnecessary columns

Prepared final dataset for model training

**4. Machine Learning Models**

The following models were trained:

Model	Status
Logistic Regression	✔ Trained & Saved
k-Nearest Neighbors ✔ Trained & Saved
Decision Tree	✔ Trained & Saved
Random Forest	✔ Best performance
Gradient Boosting	✔ Trained & Saved

All models were saved as:

lr.pkl

knn.pkl

dt.pkl

rf.pkl

gb.pkl

**5. Model Evaluation**

Evaluation metrics used:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Random Forest and Gradient Boosting performed the best.

**6. Streamlit Web Application**

A fully functional interactive UI was developed for real-time prediction.

**Features:**

User inputs employee details

Auto-select categorical & numeric fields

Backend encodes + scales inputs automatically

Predicts attrition using selected model

Shows clear output:

Attrition: YES (Likely to Leave)

Attrition: NO (Likely to Stay)

🔹 Run the app:
streamlit run emp_attrition.py

**7. Technologies Used**

Python

Pandas

NumPy

Scikit-learn

Streamlit

Matplotlib/Seaborn (EDA)

**8. Key Insights**

Overtime, low job satisfaction, and high workload are major attrition drivers

Younger employees tend to leave more frequently

Lower income groups show higher attrition

Certain job roles are more sensitive to attrition

**9. How to Use This Project**

Clone repository

Install requirements

Place dataset in project folder

Run Streamlit app

Select employee details

Get prediction instantly

 **10. Future Enhancements**

Add SHAP or Lime explainability

Build REST API using FastAPI

Deploy online on Streamlit Cloud

Add employee retention suggestions based on prediction

**Conclusion**

This project successfully demonstrates:

✔ End-to-end ML workflow

✔ Real-time prediction using Streamlit

✔ Proper model training, saving, and deployment

✔ HR-focused analytics to reduce attrition

@ Contact: 📧 Email: sneharaje167@gmail.com

🌐 LinkedIn: https://www.linkedin.com/in/sneha-rajendiran-2427651b7
