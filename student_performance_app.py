import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Mock dataset generation
def generate_mock_data(n=200):
    np.random.seed(42)
    data = pd.DataFrame({
        'Study Hours': np.random.randint(0, 15, n),
        'Sleep Hours': np.random.randint(3, 10, n),
        'Attendance Rate': np.random.randint(50, 100, n),
        'Class Participation': np.random.randint(1, 10, n),
        'Assignments Completed': np.random.randint(1, 10, n),
        'Gadget Usage (hrs)': np.random.randint(0, 12, n),
        'Likely to Pass': np.random.randint(0, 2, n)
    })
    return data

# Load and preprocess data
data = generate_mock_data()
X = data.drop("Likely to Pass", axis=1)
y = data["Likely to Pass"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Streamlit App UI
st.title("Student Performance Predictor")

st.sidebar.header("Enter Student Details")
study_hours = st.sidebar.slider("Study Hours per Day", 0, 15, 5)
sleep_hours = st.sidebar.slider("Sleep Hours per Day", 3, 10, 7)
attendance = st.sidebar.slider("Attendance Rate (%)", 50, 100, 85)
participation = st.sidebar.slider("Class Participation (1-10)", 1, 10, 5)
assignments = st.sidebar.slider("Assignments Completed (1-10)", 1, 10, 8)
gadget_use = st.sidebar.slider("Gadget Use (hrs/day)", 0, 12, 3)

if st.button("Predict"):
    input_data = pd.DataFrame([[study_hours, sleep_hours, attendance, participation, assignments, gadget_use]],
                              columns=X.columns)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("Likely to Pass")
    else:
        st.error("Needs Improvement")

    st.subheader("Model Evaluation Metrics")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")
