import pickle

import pandas as pd
import streamlit as st
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

X = df.drop(columns=['fare_amount'])
y = df['fare_amount']

st.title('Uber Fare Price Predictor')

#PassengerCount
passenger_count = st.selectbox('No. of Passengers:',[1,2,3,4,5,6])

#year
year = st.selectbox('Select Year:',df['year'].unique())

#weekday
weekday = st.selectbox('Select Week Day:',df['weekday'].unique())

#weekday
mq = st.selectbox('Select Month Quarter:',sorted(df['month_quarter'].unique()))

#weekday
hs = st.selectbox('Select Hour Segments:',sorted(df['hour_segments'].unique()))

dist = st.text_input('Enter Approx. Distance According to you:')

btn = st.button('Predict Price')

if btn:
    query = pd.DataFrame(data=[[passenger_count,year,weekday,mq,hs,dist]],columns=X.columns)

    st.title('Predicted Fare Amount is:' +str(round(pipe.predict(query)[0],2)))