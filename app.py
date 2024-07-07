import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression


st.write("""
# Customer Churn Prediction App

This app predicts the **Customer Churn!**

Data obtained from the [kaggle datasets](https://www.kaggle.com/datasets/rashadrmammadov/customer-churn-dataset) on Kaggle by Rashad Mammadov.
""")

st.sidebar.header('User Input Features')

def user_input_features():
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider('Tenure', 0, 72, 20)
    
    with col2:
        monthlyCharges = st.slider('Monthly Charges', 0, 150, 50)
    gender = st.sidebar.selectbox('Gender',('Female', 'Male'))
    seniorCitizen = st.sidebar.selectbox('Senior Citizen',('No', 'Yes'))
    partner = st.sidebar.selectbox('Partner',('No', 'Yes'))
    dependents = st.sidebar.selectbox('Dependents',('No', 'Yes'))
    
    phoneService = st.sidebar.selectbox('Phone Service',('No', 'Yes'))
    multipleLines = st.sidebar.selectbox('Multiple Lines',('No', 'Yes', 'No phone service'))
    internetService = st.sidebar.selectbox('Internet Service',('DSL', 'Fiber optic', 'No'))
    onlineSecurity = st.sidebar.selectbox('Online Security',('No', 'Yes', 'No internet service'))
    onlineBackup = st.sidebar.selectbox('Online Backup',('No', 'Yes', 'No internet service'))
    techSupport = st.sidebar.selectbox('Tech Support',('No', 'Yes', 'No internet service'))
    paperlessBilling = st.sidebar.selectbox('Paperless Billing',('No', 'Yes'))
    paymentMethod = st.sidebar.selectbox('Payment Method',('Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'))
    
    data = {'Gender': gender,
            'SeniorCitizen': seniorCitizen,
            'Partner': partner,
            'Dependents': dependents,
            'Tenure': tenure,
            'PhoneService': phoneService,
            'MultipleLines': multipleLines,
            'InternetService': internetService,
            'OnlineSecurity': onlineSecurity,
            'OnlineBackup': onlineBackup,
            'TechSupport': techSupport,
            'PaperlessBilling': paperlessBilling,
            'PaymentMethod': paymentMethod,
            'MonthlyCharges': monthlyCharges}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()


churn_raw = pd.read_csv('churn_cleaned.csv')
churn = churn_raw.drop(columns=['Churn'])
df = pd.concat([input_df,churn],axis=0)

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['SeniorCitizen'] = label_encoder.fit_transform(df['SeniorCitizen'])
df['Partner'] = label_encoder.fit_transform(df['Partner'])
df['Dependents'] = label_encoder.fit_transform(df['Dependents'])
df['PhoneService'] = label_encoder.fit_transform(df['PhoneService'])
df['MultipleLines'] = label_encoder.fit_transform(df['MultipleLines'])
df['InternetService'] = label_encoder.fit_transform(df['InternetService'])
df['OnlineSecurity'] = label_encoder.fit_transform(df['OnlineSecurity'])
df['OnlineBackup'] = label_encoder.fit_transform(df['OnlineBackup'])
df['TechSupport'] = label_encoder.fit_transform(df['TechSupport'])
df['PaperlessBilling'] = label_encoder.fit_transform(df['PaperlessBilling'])
df['PaymentMethod'] = label_encoder.fit_transform(df['PaymentMethod'])

df = df[:1]

st.image("churn_image.png")

load_clf = pickle.load(open('churn_model_pickle.pkl', 'rb'))

prediction = load_clf.predict(df)

st.subheader('Prediction')
penguins_species = np.array(['No', 'Yes'])

if penguins_species[prediction] == 'No':
    st.write("Customer will not churn.")
elif penguins_species[prediction] == 'Yes':
    st.write("Customer will churn.")
else:
    st.write("Error.")