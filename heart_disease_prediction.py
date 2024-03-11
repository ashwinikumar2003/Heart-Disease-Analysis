import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

heart = pd.read_csv(r"heart-disease.csv")
st.write('''
# Simple Heart Disease Classifier
this app classifies heart disease based upon certain parameters
''')
st.sidebar.header("User input parameters")

def user_input():
    age = st.sidebar.slider("Age", 29, 77, 40)
    sex_mapping = {"Male": 1, "Female": 0}
    sex = sex_mapping[st.sidebar.radio("Gender", ("Male", "Female"))]

    chol = st.sidebar.slider("Cholesterol", 100, 600, 300)

    trestbps = st.sidebar.slider("Resting BP", 90, 200, 140)

    cp_options = {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3}
    cp = cp_options[st.sidebar.selectbox("Chest Pain", ("typical angina", "atypical angina", "non-anginal pain", "asymptomatic"))]

    fbs_options = {"Yes":1, "No":0}
    fbs = fbs_options[st.sidebar.radio("Fasting Blood Sugar (> 120 mg/dl)", ("Yes", "No"))]

    rest_ecg_options = {"left ventricular hypertrophy": 0, "normal": 1, "ST-T wave abnormality": 2}
    rest_ecg = rest_ecg_options[st.sidebar.selectbox("Resting ECG results",
                                    ("left ventricular hypertrophy", "normal", "ST-T wave abnormality"))]

    thalach = st.sidebar.slider("Maximum Heart Rate", 70, 200, 120)

    exang_options={"Yes":1, "No":2}
    exang = exang_options[st.sidebar.radio("Exercise Induced Angina", ("Yes", "No"))]

    oldpeak = st.sidebar.slider("ST depression induced by exercise", 0.0, 6.5, 4.0)

    slope_options = {"downsloping": 0, "flat": 1, "upsloping": 2}
    slope = slope_options[st.sidebar.selectbox("Slope of the peak exercise ST segment", ("downsloping", "flat", "upsloping"))]

    ca = st.sidebar.slider("The number of major vessels", 0, 3, 2)

    thal_options = {"null": 0, "fixed defect": 1, "normal flow": 2, "reversible defect": 3}
    thal = thal_options[st.sidebar.selectbox("Thalassemia", ("null", "fixed defect", "normal flow", "reversible defect"))]

    data = {'age': age, 'sex': sex, 'cp':cp, 'trestbps':trestbps, 'chol': chol, 'fbs': fbs, 'restecg':rest_ecg, 'thalach':thalach, 'exang':exang, 'oldpeak':oldpeak, 'slope':slope, 'ca':ca, 'thal':thal}
    features = pd.DataFrame(data, index=[0])
    return features

df=user_input()
st.subheader("User Input Parameters")
st.write(df)



clf = RandomForestClassifier(n_estimators=100)

x = heart.drop('target', axis=1)
y = heart['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7)

clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
y_pred = clf.predict(df)
probability = clf.predict_proba(df)

pred = {1:"Yes", 0:"No"}
p=pred[y_pred[0]]
st.subheader("Is Heart Disease Present?")
st.write(p)
st.write(clf.score(x_test, y_test))
