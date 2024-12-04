
import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('/Users/arthe/disease/disease-predictor-app/lung_cancer_model.pkl', 'rb'))

def user_input_features():
    st.sidebar.header('User Input Parameters')
    
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 21, 87, 50)
    smoking = st.sidebar.selectbox('Smoking', ('No', 'Yes'))
    yellow_fingers = st.sidebar.selectbox('Yellow Fingers', ('No', 'Yes'))
    anxiety = st.sidebar.selectbox('Anxiety', ('No', 'Yes'))
    peer_pressure = st.sidebar.selectbox('Peer Pressure', ('No', 'Yes'))
    chronic_disease = st.sidebar.selectbox('Chronic Disease', ('No', 'Yes'))
    fatigue = st.sidebar.selectbox('Fatigue', ('No', 'Yes'))
    allergy = st.sidebar.selectbox('Allergy', ('No', 'Yes'))
    wheezing = st.sidebar.selectbox('Wheezing', ('No', 'Yes'))
    alcohol_consuming = st.sidebar.selectbox('Alcohol Consuming', ('No', 'Yes'))
    coughing = st.sidebar.selectbox('Coughing', ('No', 'Yes'))
    shortness_of_breath = st.sidebar.selectbox('Shortness of Breath', ('No', 'Yes'))
    swallowing_difficulty = st.sidebar.selectbox('Swallowing Difficulty', ('No', 'Yes'))
    chest_pain = st.sidebar.selectbox('Chest Pain', ('No', 'Yes'))

    
    binary_mapping = {'No': 1, 'Yes': 2}
    gender_encoded = 1 if gender == 'Male' else 0

    data = np.array([[
        gender_encoded, 
        age, 
        binary_mapping[smoking], 
        binary_mapping[yellow_fingers], 
        binary_mapping[anxiety],
        binary_mapping[peer_pressure], 
        binary_mapping[chronic_disease], 
        binary_mapping[fatigue], 
        binary_mapping[allergy], 
        binary_mapping[wheezing], 
        binary_mapping[alcohol_consuming], 
        binary_mapping[coughing], 
        binary_mapping[shortness_of_breath], 
        binary_mapping[swallowing_difficulty], 
        binary_mapping[chest_pain]
    ]])
    
    return data

st.title('Lung Cancer Prediction')

input_data = user_input_features()
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.subheader('Prediction')
st.write('Cancer' if prediction[0] == 1 else 'No Cancer')

st.subheader('Prediction Probability')
class_labels = ['No Cancer', 'Cancer']
proba_df = pd.DataFrame(prediction_proba, columns=class_labels)
st.table(proba_df.style.hide(axis="index"))

