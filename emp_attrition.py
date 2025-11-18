import pandas as pd
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# LOAD DATASET

df = pd.read_csv("emp_attrition.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

encoders = {}

for col in df.columns:
    if col != "Attrition" and df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# SPLIT DATA
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALE DATA
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier()
dt_model = DecisionTreeClassifier()
gb_model = GradientBoostingClassifier()

lr_model.fit(x_train_scaled, y_train)
rf_model.fit(x_train_scaled, y_train)
dt_model.fit(x_train_scaled, y_train)
gb_model.fit(x_train_scaled, y_train)

# SAVE MODELS
with open("lr_model.pkl", "wb") as f: pickle.dump(lr_model, f)
with open("rf_model.pkl", "wb") as f: pickle.dump(rf_model, f)
with open("dt_model.pkl", "wb") as f: pickle.dump(dt_model, f)
with open("gb_model.pkl", "wb") as f: pickle.dump(gb_model, f)

# LOAD ENCODERS, SCALER, AND MODELS
encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

lr_model = pickle.load(open("lr_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
dt_model = pickle.load(open("dt_model.pkl", "rb"))
gb_model = pickle.load(open("gb_model.pkl", "rb"))

# STREAMLIT APP
st.set_page_config(page_title="Employee Attrition Prediction Dashboard", layout="wide")
st.title("Employee Attrition Prediction Dashboard")
st.markdown("Predict employee attrition risk using trained ML models.")
st.sidebar.header("Enter Employee Details")


df_sample = pd.read_csv("emp_attrition.csv").drop(columns=["Unnamed: 0"], errors="ignore")
df_sample = df_sample.drop("Attrition", axis=1)

user_inputs = {}

for col in df_sample.columns:
    if col in encoders:  # categorical
        categories = list(encoders[col].classes_)
        user_inputs[col] = [st.sidebar.selectbox(col, categories)]
    else:  # numeric
        default = float(df_sample[col].median())
        user_inputs[col] = [st.sidebar.number_input(col, value=default)]

# Convert to DataFrame
data = pd.DataFrame(user_inputs)

# Apply encoders
for col in data.columns:
    if col in encoders:
        data[col] = encoders[col].transform(data[col])

# Scale numeric columns
scaled_data = scaler.transform(data)

# MODEL SELECTION
model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
)

model_map = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
}

model = model_map[model_choice]

# PREDICTION
if st.sidebar.button("Predict Attrition"):
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.error("Employee is likely to leave the company (Attrition: YES)")
    else:
        st.success("Employee is likely to stay (Attrition: NO)")

