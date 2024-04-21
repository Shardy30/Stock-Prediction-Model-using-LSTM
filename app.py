import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from keras.models import load_model
import streamlit as st
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

yf.pdr_override()

start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

st.title('Stock Prediction')
user_input = st.text_input('Enter Stock Name', 'AAPL')
df = wb.get_data_yahoo(user_input, start_date, end_date)

# Data
st.subheader('Data from 2020-2023')
st.write(df.describe())

# Graphs
st.subheader('Closing Price vs Time (with 100 moving average)')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time (with 100 and 200 moving average)')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

data_training = pd.DataFrame(df["Close"][0:int(len(df) * 0.7)])
data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.7):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

model = load_model("spm.h5")

# Testing
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i, 0])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_predicted = model.predict(x_test)

# Reverse scaling
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))

# Final Visualization
st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
