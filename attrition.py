import pandas as pd
import streamlit as st
import pickle

#loading models
attrition_model = {
    "Logistic Regression": pickle.load(open("lr.pkl", "rb")),
    "Random Forest": pickle.load(open("rf.pkl", "rb")),
    "Decision Tree": pickle.load(open("dt.pkl", "rb")),
    "Gradient Boosting": pickle.load(open("gb.pkl", "rb")) 
}

# Scaler and encoders
scaler = pickle.load(open("scaler.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))
le_gender = pickle.load(open("le_gender.pkl", "rb"))       # For Gender
le_overtime = pickle.load(open("le_overtime.pkl", "rb"))   # For OverTime
columns = pickle.load(open("columns.pkl", "rb"))           # Columns used during training

# Streamlit App
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("Employee Attrition Prediction Dashboard")

# Model selection
selected_model = st.sidebar.selectbox("Choose Model", list(attrition_model.keys()))
model = attrition_model[selected_model]

#input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
businesstravel = st.selectbox("BusinessTravel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
dailyrate = st.number_input("DailyRate", min_value=100, max_value=15000, value=1000)
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
distancefromhome = st.number_input("DistanceFromHome", min_value=1, max_value=50, value=10)
education = st.selectbox("Education", [1,2,3,4,5])
educationfield = st.selectbox("EducationField", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
environmentsatisfaction = st.selectbox("EnvironmentSatisfaction", [1,2,3,4])
gender = st.selectbox("Gender", ["Male","Female"])
hourlyrate = st.number_input("HourlyRate", min_value=30, max_value=200, value=60)
jobinvolvement = st.selectbox("JobInvolvement", [1,2,3,4])
joblevel = st.selectbox("JobLevel", [1,2,3,4,5])
jobrole = st.selectbox("JobRole", ["Sales Executive", "Research Scientist", "Laboratory Technician", 
                                   "Manufacturing Director", "Healthcare Representative", "Manager", 
                                   "Sales Representative", "Research Director", "Human Resources"])
jobsatisfaction = st.selectbox("JobSatisfaction", [1,2,3,4])
maritalstatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
monthlyincome = st.number_input("MonthlyIncome")
monthlyrate = st.number_input("MonthlyRate")
numcompaniesworked = st.number_input("NumCompaniesWorked", min_value=0, max_value=20, value=1)
overtime = st.selectbox("OverTime", ["Yes","No"])
percentsalaryhike = st.number_input("PercentSalaryHike", min_value=0, max_value=100, value=15)
performancerating = st.selectbox("PerformanceRating", [1,2,3,4])
relationshipsatisfaction = st.selectbox("RelationshipSatisfaction", [1,2,3,4])
stockoptionlevel = st.selectbox("StockOptionLevel", [0,1,2,3])
totalworkingyears = st.number_input("TotalWorkingYears", min_value=0, max_value=40, value=5)
trainingtimeslastyear = st.number_input("TrainingTimesLastYear", min_value=0, max_value=10, value=2)
worklifebalance = st.selectbox("WorkLifeBalance", [1,2,3,4])
yearsatcompany = st.number_input("YearsAtCompany", min_value=0, max_value=40, value=3)
yearsincurrentrole = st.number_input("YearsInCurrentRole", min_value=0, max_value=20, value=2)
yearssincelastpromotion = st.number_input("YearsSinceLastPromotion", min_value=0, max_value=15, value=1)
yearswithcurrmanager = st.number_input("YearsWithCurrManager", min_value=0, max_value=20, value=2)


# Creating input dataframe
input_df = pd.DataFrame([{
    'Age': age,
    'BusinessTravel': businesstravel,
    'DailyRate': dailyrate,
    'Department': department,
    'DistanceFromHome': distancefromhome,
    'Education': education,
    'EducationField': educationfield,
    'EnvironmentSatisfaction': environmentsatisfaction,
    'Gender': gender,
    'HourlyRate': hourlyrate,
    'JobInvolvement': jobinvolvement,
    'JobLevel': joblevel,
    'JobRole': jobrole,
    'JobSatisfaction': jobsatisfaction,
    'MaritalStatus': maritalstatus,
    'MonthlyIncome': monthlyincome,
    'MonthlyRate': monthlyrate,
    'NumCompaniesWorked': numcompaniesworked,
    'OverTime': overtime,
    'PercentSalaryHike': percentsalaryhike,
    'PerformanceRating': performancerating,
    'RelationshipSatisfaction': relationshipsatisfaction,
    'StockOptionLevel': stockoptionlevel,
    'TotalWorkingYears': totalworkingyears,
    'TrainingTimesLastYear': trainingtimeslastyear,
    'WorkLifeBalance': worklifebalance,
    'YearsAtCompany': yearsatcompany,
    'YearsInCurrentRole': yearsincurrentrole,
    'YearsSinceLastPromotion': yearssincelastpromotion,
    'YearsWithCurrManager': yearswithcurrmanager
}])
 
#review target column
if 'Attrition' in input_df.columns:
    input_df = input_df.drop(['Attrition'], axis=1)

#apply le
input_df["Gender"] = le_gender.transform(input_df["Gender"])
input_df["OverTime"] = le_overtime.transform(input_df["OverTime"])

#apply ohe
ohe_cols = ['BusinessTravel','Department','EducationField','JobRole','MaritalStatus']
ohe_df = pd.DataFrame(ohe.transform(input_df[ohe_cols]), columns=ohe.get_feature_names_out(ohe_cols))
input_df = pd.concat([input_df.drop(ohe_cols, axis=1), ohe_df], axis=1)

# Reindex to ensure all columns are present
input_df = input_df.reindex(columns=columns, fill_value=0)

# Prediction
if st.sidebar.button("Predict Attrition"):
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.error("🚨 Employee is likely to leave the company (Attrition: YES)")
    else:
        st.success("✅ Employee is likely to stay (Attrition: NO)")

