# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 08:44:41 2021

@author: Bunnyyyyyyy
"""

#core pksgs
import streamlit as st
import os
import joblib
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True) 
import matplotlib.pyplot as plt



#EDA pkgs

import pandas as pd
import numpy as np






#Data viz pkgs

#import matplotib.pyplot as plt
#import matplotlib
#matplot.use('Agg')
import seaborn as sns

### load dataset

@st.cache
def load_data(dataset):
    df=pd.read_csv(dataset)
    return df


### load model
    
def load_model_prediction(model_file):
    loaded_model=joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model
    


### creating diconery method for values
gender_label= {"Male": 0, "Female": 1, "Other":2}
work_type_label= {'Private': 0, 'Self-employed': 1, 'Govt_job':2, 'children':3, 'Never_worked':4}
Residence_type_label={'Urban': 0, 'Rural': 1}
smoking_status_label={'formerly smoked': 0, 'never smoked': 1, 'smokes':2, 'Unknown':3}
ever_married_label={'No': 0, 'Yes': 1} 
hypertension_label={'No': 0, 'Yes': 1}
heart_disease_label={'No': 0, 'Yes': 1}
strok_label={0:" No Risk to get Stroke", 1:"Risk of getting Stroke"}

data=load_data('healthcare-dataset-stroke-data.csv')



age_min=data.age.min()
age_max=data.age.max()


avg_glucose_level_min=data.avg_glucose_level.min()
avg_glucose_level_max=data.avg_glucose_level.max()


bmi_min=data.bmi.min()
bmi_max=data.bmi.max()







## get keys

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val==key:
            return value
## find the key from the dictonary
            
def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val==key:
            return key
    



    




def main():
    
    """ main """
    
    #menu
    
    menu=["Home","Project Data","Data Visualization",'Prediction',"Developer"]
    choices=st.sidebar.selectbox("Menu",menu)
    

    if choices=="Home":
        st.title("Stroke Prediction Analysis")
        st.subheader("Build with Streamlit")
        image = Image.open('images/stroke - Copy.jpg')
        st.image(image, caption='Developed by Sai')
        st.text('According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.')
        st.text('')
        st.text('This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.')
        st.text('')
        st.subheader('Attribute Information ')
        st.text('')
        st.text('1) id: unique identifier')
        st.text('2) gender: "Male", "Female" or "Other"')
        st.text('3) age: age of the patient')
        st.text("4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension")
        st.text("5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease")
        st.text('6) ever_married: "No" or "Yes"')
        st.text('7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"')
        st.text('8) Residence_type: "Rural" or "Urban"')
        st.text('9) avg_glucose_level: average glucose level in blood')
        st.text('10) bmi: body mass index')
        st.text('11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*')
        st.text('12) stroke: 1 if the patient had a stroke or 0 if not')
        st.text('')
        st.text('*Note: "Unknown" in smoking_status means that the information is unavailable for this patient')


    if choices=="Project Data":
        st.title("Stroke Prediction Analysis")
        st.subheader("Build with Streamlit")
        st.subheader("About the Project and its  Details")
        st.text('')
        st.text('Sample Data....')
        
        data=load_data('healthcare-dataset-stroke-data.csv')
        st.dataframe(data.head(10))
        
        
        if st.checkbox("Show Summary Metrics"):
            st.write(data.describe())
            
        


    if choices=="Data Visualization":
        st.title("Stroke Prediction Analysis")
        st.subheader("Build with Streamlit")
        st.subheader("Visualizing data")
        data1=load_data('healthcare-dataset-stroke-data.csv')
        st.text('')
        st.checkbox("Visualizing data with gender with type of smokers")
        st.write(sns.countplot(x='gender',hue='smoking_status',data=data1))
        st.text('')
        st.write(px.pie(data1, values='gender', names='smoking_status'))
        y = np.array([data1[data1['smoking_status']==0].count().values[0],data1[data1['smoking_status']==1].count().values[0],data1[data1['smoking_status']==2].count().values[0],data1[data1['smoking_status']==3].count().values[0]])
        mylabels = ["formerly smoke", "never smoked", "smokes", "Unknown"]
        plt.pie(y, labels = mylabels)
        st.write(plt.legend(title = "smokers types with male and female:"))
         
        
        
        
        
        
        
        
        
    if choices=="Prediction":
        st.title("Stroke Prediction Analysis")
        st.subheader("Build with Streamlit")
        st.subheader("Prediction")

        
        gender=st.selectbox("Select the Gender",tuple(gender_label.keys()))
        age=st.number_input("Select the  Age of a person",age_min,age_max)
        hypertension=st.selectbox("Select the hypertension",tuple(hypertension_label.keys()))
        heart_disease=st.selectbox("Select the heart_disease",tuple(heart_disease_label.keys()))
        ever_married=st.selectbox("Select the ever_married",tuple(ever_married_label.keys()))
        work_type=st.selectbox("Select the work_type",tuple(work_type_label.keys()))
        Residence_type=st.selectbox("Select the Residence_type",tuple(Residence_type_label.keys()))
        avg_glucose_level=st.number_input("Select the avg_glucose_level",avg_glucose_level_min,avg_glucose_level_max)
        bmi=st.number_input("Select the bmi_level",bmi_min,bmi_max)
        smoking_status=st.selectbox("Select the smoking_status",tuple(smoking_status_label.keys()))
        
        ### encoding
        
        gender_v= get_value(gender,gender_label)
        hypertension_v= get_value(hypertension,hypertension_label)
        heart_disease_v= get_value(heart_disease,heart_disease_label)
        ever_married_v= get_value(ever_married,ever_married_label)
        work_type_v= get_value(work_type,work_type_label)
        Residence_type_v= get_value(Residence_type,Residence_type_label)
        smoking_status_v= get_value(smoking_status,smoking_status_label)
        
        
        pretty_data={
                "gender":gender,
                "age":age,
                "hypertension":hypertension,
                "heart_disease":heart_disease,
                "ever_married":ever_married,
                "work_type":work_type,
                "Residence_type":Residence_type,
                "avg_glucose_level":avg_glucose_level,
                "bmi":bmi,
                "smoking_status":smoking_status
                }
        st.subheader("Options Selected")
        st.json(pretty_data)
        
        st.subheader("Encoded data")
        encoding_data=[gender_v,age,hypertension_v,heart_disease_v,
                       ever_married_v,work_type_v,Residence_type_v,
                       avg_glucose_level,bmi,smoking_status_v]
        st.write(encoding_data)
        
        # input data should be 2d not one 1d so we convert here
        prep_encoding_data=np.array(encoding_data).reshape(1,-1)
        
        model_choice=st.selectbox("Model_Choice",["Logistic_Regression","Support_Vector_Machines"])
        if st.button("Evaluate"):
            if model_choice=="Logistic_Regression":
                predictor=load_model_prediction("models/healthcare-stroke-prediction_Logistic.pkl")
                prediction=predictor.predict(prep_encoding_data)
                
            elif model_choice=="Support_Vector_Machines":
                predictor=load_model_prediction("models/healthcare-stroke-prediction_SVC.pkl")
                prediction=predictor.predict(prep_encoding_data)
                
                
            final_result=get_value(prediction,strok_label)
            st.success(final_result)
                
                
                
    
        
    if choices=="Developer":
        st.title("Stroke Prediction Analysis")
        st.subheader("Sairamdgr8 -- An Aspiring Full Stack Data Engineer")
        image = Image.open('images/dev - Copy.jpg')
        st.image(image, caption='Developed by Sai')
        st.text('')
        st.subheader('Connect  me... ')
        st.subheader('https://www.linkedin.com/in/sairam-p-l/')
        st.subheader('https://medium.com/@sairamdgr8')
        st.subheader('https://www.facebook.com/bunnydgr8')
        
        
    #if choices=='Testing':
        #st.set_page_config(page_title="Ex-stream-ly Cool App",page_icon="🧊",layout="wide",initial_sidebar_state="expanded")
        
  

if __name__=='__main__':
    main()
    
    