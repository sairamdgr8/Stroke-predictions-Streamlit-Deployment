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
import altair as alt
import plotly.figure_factory as ff
from plotly.subplots import make_subplots




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
        st.subheader("Visualizing data with respective Dataset")
        # with object values 
        data1=load_data('healthcare-dataset-stroke-data.csv')
        # without object values
        #data_conv=load_data('converted dataframe.csv')
        st.text('Visualizing data with All genders with type of smokers')
        #st.checkbox("Visualizing data with gender with type of smokers")
        #st.write(sns.countplot(x='gender',hue='smoking_status',data=data1))
        #st.text('')
        #st.write(px.pie(data1, values='gender', names='smoking_status'))
        
        
        #Making the Simple Bar Chart
        #### people who smokes
        #smoking_status_label={'formerly smoked': 0, 'never smoked': 1, 'smokes':2, 'Unknown':3}
        smoking_data = pd.DataFrame({'total population who smokes' : ["formerly smoke", "never smoked", "smokes", "Unknown"],
                                     'both male and female': np.array([data1[data1['smoking_status']=='formerly smoked'].count().values[0],data1[data1['smoking_status']=='never smoked'].count().values[0],data1[data1['smoking_status']=='smokes'].count().values[0],data1[data1['smoking_status']=='Unknown'].count().values[0]])
                                     })
        
        st.write(alt.Chart(smoking_data).mark_bar().encode(
            # Mapping the Website column to x-axis
            y='total population who smokes',
            # Mapping the Score column to y-axis
            x='both male and female'))
        
        ################################
        
        st.text('Visualizing data with BMI levels with Male')
        filter_male=data1['gender']=='Male'
        wrt_male=data1.where(filter_male)
        st.bar_chart(wrt_male['bmi'])
        
        #############################
        
        
        
        #Making the line Chart wrt male
        ## male bmi levels
        
        #wrt_male_bmi=wrt_male[['bmi']]
        wrt_male_data=pd.DataFrame(wrt_male['bmi'],columns=['bmi with males'])
        st.line_chart(wrt_male_data)
        
        
        #Making the line Chart wrt male
        ## male bmi levels
        #not working
        
        #filter_male=data_conv['gender']==0
        #data_conv.where(filter_male, inplace = True)
        #hist_data_male = [data_conv['bmi'],data_conv['avg_glucose_level']]
        #group_labels_male = ['bmi', 'glucose']
        #fig_male = ff.create_distplot(hist_data_male, group_labels_male, bin_size=[10, 25])
        #st.plotly_chart(fig_male, use_container_width=True)
        
        
        st.subheader('Visualizing data with BMI levels with Male')
        #Making the plot Chart wrt male
        ## male bmi levels     
        data_conv=pd.read_csv('converted dataframe.csv')
        filter_male=data_conv['gender']==0
        wrt_male=data_conv.where(filter_male)
        arr = wrt_male['bmi']
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        plt.xlabel('BMI Levels')
        plt.ylabel('with respective males')
        st.pyplot(fig)
        
        #data_conv=pd.read_csv('converted dataframe.csv')
       #filter_male=data_conv['gender']==0
        
        arr = wrt_male['avg_glucose_level']
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        plt.xlabel('Avg_Glucose_Levels')
        plt.ylabel('with respective males')
        st.pyplot(fig)
        
        
        filter_female=data_conv['gender']==1
        wrt_female=data_conv.where(filter_female)
        arr = wrt_female['bmi']
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        plt.xlabel('BMI Levels')
        plt.ylabel('with respective females')
        st.pyplot(fig)
        
        
        arr = wrt_female['avg_glucose_level']
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        plt.xlabel('Avg_Glucose_Levels')
        plt.ylabel('with respective females')
        st.pyplot(fig)
        
        
        fig = px.pie(data_conv, values='gender', names='smoking_status',title='Total persons consumption smoking category')
        st.plotly_chart(fig) 
        
        
#        fig = go.Figure(data=[
#        go.Bar(name='Male', x=data1[['gender']=='Male'], y=np.array([data1[data1['smoking_status']==0].count().values[0],data1[data1['smoking_status']==1].count().values[0],data1[data1['smoking_status']==2].count().values[0],data1[data1['smoking_status']==3].count().values[0]])),
#        go.Bar(name='Female', x=data1[['gender']=='Female'], y=np.array([data1[data1['smoking_status']==0].count().values[0],data1[data1['smoking_status']==1].count().values[0],data1[data1['smoking_status']==2].count().values[0],data1[data1['smoking_status']==3].count().values[0]])),
#        go.Bar(name='Other', x=data1[['gender']=='Other'], y=np.array([data1[data1['smoking_status']==0].count().values[0],data1[data1['smoking_status']==1].count().values[0],data1[data1['smoking_status']==2].count().values[0],data1[data1['smoking_status']==3].count().values[0]]))
#        st.plotly_chart(fig)
        
        
        
        
        
        

        
        
        
        
        
     
        
        
        
            
         
        
        
        
        
        
        
        
        
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
        #st.subheader('https://www.linkedin.com/in/sairam-p-l/')
        #st.subheader('https://medium.com/@sairamdgr8')
        #st.subheader('https://www.facebook.com/bunnydgr8')
        """
        [![Linkledn Follow](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sairam-p-l/) https://www.linkedin.com/in/sairam-p-l/
        
        [![medium Follow](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@sairamdgr8)  https://medium.com/@sairamdgr8 
        
        [![facebook Follow](https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://www.facebook.com/bunnydgr8) https://www.facebook.com/bunnydgr8
        
        ![Gmail Follow](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white) sairamdgr8@gmail.com
   
        """
        
        
        
    #if choices=='Testing':
        #st.set_page_config(page_title="Ex-stream-ly Cool App",page_icon="🧊",layout="wide",initial_sidebar_state="expanded")
        
  

if __name__=='__main__':
    main()
    
    