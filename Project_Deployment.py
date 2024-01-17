# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 01:09:34 2024

@author: 002
"""

import pandas as pd 
import numpy as np
import plotly.express as px
import streamlit as st 
import joblib 
import category_encoders as ce

Model=joblib.load('Model.h5')

df=pd.read_csv('Australian_Vehicle_Prices_Data_Cleaned.csv')

st.set_page_config(layout="wide")
with st.sidebar:
    radio=st.radio('what do you wnat to choose',('Documentation','DataFrame','Charts','Prediction'))
# The Documentation Section      
if radio =='Documentation':
    st.title('Documentation')
    st.subheader('''
                This is an project to predict the Price of the Australian Vehicle Prices 
                 ''')
    st.text_area('Description:','This dataset contains the latest information on car prices in Australia for the year 2023. It covers various brands, models, types, and features of cars sold in the Australian market. It provides useful insights into the trends and factors influencing the car prices in Australia. The dataset includes information such as brand, year, model, car/suv, title, used/new, transmission, engine, drive type, fuel type, fuel consumption, kilometres, colour (exterior/interior), location, cylinders in engine, body type, doors, seats, and price. The dataset has over 16,000 records of car listings from various online platforms in Australia.')
    st.text_area('Columns:','''
       - Brand: Name of the car manufacturer
       - Year: Year of manufacture or release
       - Model: Name or code of the car model
       - Car/Suv: Type of the car (car or suv)
       - Title: Title or description of the car
       - UsedOrNew: Condition of the car (used or new)
       - Transmission: Type of transmission (manual or automatic)
       - Engine: Engine capacity or power (in litres or kilowatts)
       - DriveType: Type of drive (front-wheel, rear-wheel, or all-wheel)
       - FuelType: Type of fuel (petrol, diesel, hybrid, or electric)
       - FuelConsumption: Fuel consumption rate (in litres per 100 km)
       - Kilometres: Distance travelled by the car (in kilometres)
       - ColourExtInt: Colour of the car (exterior and interior)
       - Location: Location of the car (city and state)
       - CylindersinEngine: Number of cylinders in the engine
       - BodyType: Shape or style of the car body (sedan, hatchback, coupe, etc.)
       - Doors: Number of doors in the car
       - Seats: Number of seats in the car
       - Price: Price of the car (in Australian dollars)

''',height=500)


#the DataFrame Section
if radio=='DataFrame':
    listTabs=['DataSet Sample','DataSet Describtive Stats']
    tabs=st.tabs([s.center(1,"\t") for s in listTabs])
    #DataFrame
    with tabs[0]:
        st.subheader('Subset of the DataFrame (First 30 row)')
        st.dataframe(df.head(30),height=700)
    #Stats 
    with tabs[1]:
        st.subheader('Some Stats for Numerical Columns')
        st.dataframe(df.describe().T)
        
        
#The ChartsSection
if radio=='Charts':
    tab1, tab2 = st.tabs(['📈  Numerical Charts','📊 Categorical Charts']) 
    #Numerical Charts
    with tab1:
        col1,col2,col3=st.columns([6,0.5,6])
        with col1:
            st.subheader('Distribution of Year')
            fig=px.histogram(df['Year'])
            st.plotly_chart(fig,use_container_width=True)
        with col3:
            st.header('what the distribution of the price?')
            fig=px.histogram(df['Price'])
            st.plotly_chart(fig,use_container_width=True)
            
            
        col4,col5,col6=st.columns([6,0.5,6])
        with col4:
          st.subheader('Cylinders in Engine Cars most selling')
          fig=px.histogram(df['CylindersinEngine'].astype(str))
          st.plotly_chart(fig,use_container_width=True) 
        
        with col6:
            st.subheader('Litres in Engine Cars most selling')
            fig=px.histogram(df['LitersinEngine'].astype(str)).update_xaxes(categoryorder='total descending')
            st.plotly_chart(fig,use_container_width=True) 
        
        
        st.subheader('The impact on the Price ')
        impact=st.selectbox('selecet what you want to impact with the Price',('FuelConsumption','Doors','Seats','CylindersinEngine','LitersinEngine'))
        st.write('The impact of ',impact,' on the Price')
        fig=px.line(x=df.groupby([impact])['Price'].mean().reset_index().sort_values(by=[impact],ascending=False).astype(str)[impact],
        y=df.groupby([impact])['Price'].mean().reset_index().sort_values(by=[impact],ascending=False)['Price'],labels={'x':impact,'y':'Price'})
        st.plotly_chart(fig,use_container_width=True)
        
        
        st.subheader('The impact Combination on the Price ')
        option=st.selectbox('selecet what you want to impact with the Price',('Doors Seats','CylindersinEngine LitersinEngine'))
        option=option.split(' ')
        fig=px.line(x=df.groupby([option[0],option[1]])['Price'].mean().reset_index()[option[1]],
        y=df.groupby([option[0],option[1]])['Price'].mean().reset_index()['Price'],
        color=df.groupby([option[0],option[1]])['Price'].mean().reset_index()[option[0]],
        labels={'x':'Num of '+option[1],'y':'Price','color':'Num of '+option[0]})
        st.plotly_chart(fig,use_container_width=True)
        
        st.subheader('Scatter Matrix plot')
        scatter_mat_options=st.multiselect('What do you want to add in the scatter matrix',['Year', 'FuelConsumption', 'Kilometres', 'CylindersinEngine', 'Doors',
               'Seats', 'Price', 'LitersinEngine'])
        fig=px.scatter_matrix(df,dimensions=scatter_mat_options)
        st.plotly_chart(fig,use_container_width=True)
        
        st.subheader('Correlation of the Numeric Columns')
        fig=px.imshow(df.select_dtypes('number').corr(),text_auto=True)
        st.plotly_chart(fig,use_container_width=True)
        
    #Category Charts
    with tab2:
        col1,col2,col3=st.columns([6,0.5,6])
        with col1:
            st.subheader('The most expensive Cars')
            fig=px.histogram(x=df.groupby(['Brand & Model'])['Price'].mean().reset_index().sort_values(by=['Price'],ascending=False).head(15)['Brand & Model'],
             y=df.groupby(['Brand & Model'])['Price'].mean().reset_index().sort_values(by=['Price'],ascending=False).head(15)['Price'])
            st.plotly_chart(fig)
        
        with col3:
            st.subheader('The most cheapest Cars')
            fig=px.histogram(x=df.groupby(['Brand & Model'])['Price'].mean().reset_index().sort_values(by=['Price'],ascending=True).head(15)['Brand & Model'],
             y=df.groupby(['Brand & Model'])['Price'].mean().reset_index().sort_values(by=['Price'],ascending=True).head(15)['Price']).update_xaxes(categoryorder='total ascending')
            st.plotly_chart(fig)
        
        col4,col5,col6=st.columns([6,0.5,6])
        with col4:
            st.subheader('Color of most Cars selling')
            fig=px.histogram(df['ColourExtInt']).update_xaxes(categoryorder='total descending')
            st.plotly_chart(fig)
        
        with col6:
            st.subheader('Body Type of most Cars Selling')
            fig=px.histogram(df['BodyType']).update_xaxes(categoryorder='total descending')
            st.plotly_chart(fig)
        
        col7,col8,col9=st.columns([6,0.5,6])
        with col7:
            st.subheader('state of the most purchased cars')
            fig=px.histogram(df['State']).update_xaxes(categoryorder='total descending')
            st.plotly_chart(fig)
            
        with col9:
            st.subheader('condition of the most selling Cars')
            fig=px.pie(names=df['UsedOrNew'])
            st.plotly_chart(fig)
        
        col10,col11,col12=st.columns([6,0.5,6])
        with col10:
            st.subheader('Transmission of the most selling Cars')
            fig=px.pie(names=df['Transmission'])
            st.plotly_chart(fig)
        
        with col12:
            st.subheader('Fuel Type of the most selling Cars')
            fig=px.pie(names=df['FuelType'])
            st.plotly_chart(fig)
        
        
if radio=='Prediction':
   def Predict(brand,model,BodyType,year,Condition,Transmission,DriveType,FuelType,FuelConsumption,
               Kilometres,Color,Cylinders,Liters,Doors,Seats,State):
       
       brand_model=brand+' '+model
       
       test=pd.DataFrame(data=[[year,Condition,Transmission,DriveType,FuelType,FuelConsumption,Kilometres,Color,
                                Cylinders,BodyType,Doors,Seats,brand_model,Liters,State]],
                         columns=['Year','UsedOrNew','Transmission','DriveType','FuelType','FuelConsumption','Kilometres','ColourExtInt',
                                  'CylindersinEngine','BodyType','Doors' ,'Seats','Brand & Model','LitersinEngine','State'])
       return np.exp(Model.predict(test))
       
    
   def main():
       brand = st.selectbox('Select the Brand of the Car',sorted(df['Brand'].unique())) 
       
       model = st.selectbox('Select the Model of the Car',sorted(df[df['Brand']==brand]['Model'].unique()))
       
       BodyType=st.selectbox('Select the Body Type of the Car',df['BodyType'].unique())
       
       year = st.selectbox("Select the Year of manufacture or release",sorted(df['Year'].unique()))
            
       Condition=st.radio('Select the Condition of the Car ',df['UsedOrNew'].unique())
       
       Transmission=st.radio('Select the Transmission of the Car',df['Transmission'].unique())
       
       DriveType=st.radio('Select the Drive Type of the Car ',df['DriveType'].unique())
       
       FuelType=st.radio('Select the Fuel Type of the Car',df['FuelType'].unique())
       
       FuelConsumption=st.selectbox('Select the Fuel Consumption Per 100km ',sorted(df['FuelConsumption'].unique()))
       
       Kilometres =st.number_input('Enter how many Kilometres the car traveled')
       
       Color=st.selectbox('Select the Color of the Car',sorted(df['ColourExtInt'].unique()))
       
       Cylinders =st.selectbox('Select how many Cylinders in Engine',sorted(df['CylindersinEngine'].unique()))
       
       Liters =st.selectbox('Select how many Liters in Engine',sorted(df['LitersinEngine'].unique()))
       
       Doors=st.selectbox('Select how many Doors in the Car ',sorted(df['Doors'].unique()))
       
       Seats=st.selectbox('Select how many Seats in the Car ',sorted(df['Seats'].unique()))
       
       State=st.selectbox('Select the State ',sorted(df['State'].unique()))
       
       if st.button('Predict the Price'):
           ans=Predict(brand,model,BodyType,year,Condition,Transmission,DriveType,FuelType,FuelConsumption,
                       Kilometres,Color,Cylinders,Liters,Doors,Seats,State)
           st.write(ans)
    
   
   main() 
   
   
   
   
   