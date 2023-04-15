import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

data = pd.read_csv('data.csv')

st.caption('Copyright © @muhsandii 2023')

sns.set_style('whitegrid')
sns.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

data['datetime'] = pd.to_datetime(data['datetime'])
daily_rentals = data.resample('D', on='datetime')['total_count'].sum()

st.title('Bike Rental Dashboard :bar_chart:')

# Add logo to the side bar
st.sidebar.image('bike.png')

if st.sidebar.button('Show Daily Rental'):
    st.subheader('Daily Bike Rental Frequency')
    chart = sns.lineplot(data=daily_rentals)
    chart.set(xlabel='Date', ylabel='Rental Count', title='Daily Bike Rental')
    st.pyplot()

if st.sidebar.button('Show Hourly Rental by Seasons'):
    st.subheader('Hourly Rental by Seasons')
    fig,ax = plt.subplots()
    sns.pointplot(data=data[['hour',
                               'total_count',
                               'season']],
                  x='hour',
                  y='total_count',
                  hue='season',
                  ax=ax)
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Rental Count')
    st.pyplot()

if st.sidebar.button('Show Hourly Rental by Weekday'):
    st.subheader('Hourly Rental by Weekday')
    fig,ax = plt.subplots()
    sns.pointplot(data=data[['hour',
                               'total_count',
                               'weekday']],
                  x='hour',
                  y='total_count',
                  hue='weekday',
                  ax=ax)
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Rental Count')
    st.pyplot()

if st.sidebar.button('Show Monthly Rental'):
    st.subheader('Monthly Rental Distribution')
    fig,ax = plt.subplots()
    sns.barplot(data=data[['month',
                               'total_count']],
                  x='month',
                  y='total_count',
                  ax=ax)
    ax.set_xlabel('Month')
    ax.set_ylabel('Rental Count')
    st.pyplot()

if st.sidebar.button('Show Seasonal Rental'):
    st.subheader('Seasonal Rental')
    fig,ax = plt.subplots()
    sns.barplot(data=data[['season',
                               'total_count']],
                  x='season',
                  y='total_count',
                  ax=ax)
    ax.set_xlabel('Season')
    ax.set_ylabel('Rental Count')
    st.pyplot()

st.set_option('deprecation.showPyplotGlobalUse', False)