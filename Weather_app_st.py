import pandas as pd
import streamlit as st
from neuralprophet import NeuralProphet
import plotly.express as px
import altair as alt
from PIL import Image

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#loading the data
@st.cache_data
def load_data():
    weather_data = pd.read_csv('cities_of_venezuela_mod.csv', parse_dates=[['year','month']])
    return weather_data

weather_data= load_data()  

#image in the sidebar
image = Image.open('venezuela_picture-01.png')
st.sidebar.image(image)


#Data cleaning
weather_data.rename(columns= {'year_month':'date', 't2m': 'Temperature', 't2m_max':'Temp Max', 't2m_min': 'Temp Min',
                      'prec_acum':'Precipitation'}, inplace= True)

print(weather_data.tail())

print(weather_data['city'].unique())  


weather_data.replace('San CristÃ³bal', 'San Cristóbal', inplace=True)
weather_data.replace('CumanÃ¡', 'Cumaná', inplace=True)
weather_data.replace('MaturÃ\xadn', 'Maturín', inplace=True)

#verificando los valores unicos de la columna city
print(weather_data['city'].unique())  

weather_data['Year']= weather_data['date'].dt.year
weather_data['Month']= weather_data['date'].dt.month
print(weather_data.head())

#creando los graficos de temperatura

cities= weather_data['city'].unique()

#making the app
st.sidebar.title(':blue[Weather Forecating in the citie of Venezuela]')
sidebar_box=  st.sidebar.selectbox('SELECT CITY', cities)
sidebar_slider= st.sidebar.slider('SELECT YEAR:', 1981, 2021)
st.sidebar.markdown('''
---
Created by Elianneth Cabrera
''')


temperature_city = (
    weather_data[
        (weather_data.Year <= sidebar_slider) &
        (weather_data.city == sidebar_box) &
        (weather_data.city.isin(cities))
    ]    
    .groupby(['city', 'Year'])['Temperature'].mean()
    .to_frame()
    .reset_index()
    .sort_values(by='Year')  
    .reset_index(drop=True)
)

#creating columns for the plots
col1, col2 = st.columns(2)

with col1:
#Let's create the interactivate plot
    st.subheader('Temperature over the years')

    chart= alt.Chart(temperature_city).mark_line().encode(
        x='Year',
        y=alt.Y('Temperature', scale=alt.Scale(domain=[temperature_city['Temperature'].min(),
                                                    temperature_city['Temperature'].max()])))

    chart = chart.properties(
        width=620,  # Set the width of the chart
        height=410  # Set the height of the chart
    )
    st.altair_chart(chart)

#Now we are going to create thet precipitation chart
with col2:
    precipitation_city = (
        weather_data[
            (weather_data.Year <= sidebar_slider) &
            (weather_data.city == sidebar_box) &
            (weather_data.city.isin(cities))
        ]    
        .groupby(['city', 'Year'])['Precipitation'].mean()
        .to_frame()
        .reset_index()
        .sort_values(by='Year')  
        .reset_index(drop=True)
    )

    #Let's create the interactivate plot
    st.subheader('Precipitation over the years')
    st.line_chart(data=precipitation_city, x='Year', y='Precipitation')


#metrics
col3, col4 = st.columns(2)
#col1
group_year_temp= temperature_city.groupby('Year')['Temperature'].mean()
max_temp_year = group_year_temp.sort_values(ascending=False).index[0]
col3.metric(":blue[**Most hot year**]", max_temp_year)

#col2
group_year_prec= precipitation_city.groupby('Year')['Precipitation'].mean()
max_prec = group_year_prec.sort_values(ascending=False).index[0]
col4.metric(":blue[**Most rainy year**]", max_prec)

#Correlation between Temperature and Precipitation over in the years

fig_2 = px.scatter(temp_prec_city, x="Temperature", y="Precipitation")
st.subheader('Temperature vs. Precipitation')
st.plotly_chart(fig_2)

######FORECASTING######

# Rename columns
year_pred = st.slider('Years of prediction:', 1, 4)
period = year_pred * 365

weather_data.rename(columns={'date': 'ds', 'Temperature': 'y'}, inplace=True)


# Filter the dataset based on the selected city
city_data = weather_data[weather_data['city'] == sidebar_box].reset_index(drop=True)

df_train= city_data[['ds','y']]

# Create a NeuralProphet model
model = NeuralProphet()

# Fit the model on the data
model.fit(df_train, freq='D')

# Make a forecast
future = model.make_future_dataframe(df_train, periods=period, n_historic_predictions=len(city_data))
forecast = model.predict(future)

# Plot the forecast             
st.title(f'Forecasting for {sidebar_box}')

# Plot the forecast with interactivity using plotly
fig_3 = model.plot(forecast)
st.plotly_chart(fig_3)