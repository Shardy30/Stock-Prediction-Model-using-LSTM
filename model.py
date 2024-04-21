import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Override pandas_datareader's default method to use yfinance
yf.pdr_override()

start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# Fetching data
df = wb.get_data_yahoo('AAPL', start_date, end_date)
df = df.reset_index()
df = df.drop(['Date', 'Adj Close'], axis=1)

# Data preprocessing
data_training = df["Close"][:int(len(df) * 0.7)]  # Using 70% of data for training
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_scaled = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

x_train, y_train = [], []
# Creating sequences of 100 days for training
for i in range(100, len(data_training_scaled)):
    x_train.append(data_training_scaled[i-100:i, 0])  # Input features
    y_train.append(data_training_scaled[i, 0])        # Target values
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape for LSTM input

# Model
model = Sequential()

# First LSTM layer with 50 units, return_sequences=True for returning sequences
model.add(LSTM(units=50, return_sequences=True, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting

# Second LSTM layer with 60 units
model.add(LSTM(units=60, return_sequences=True, activation='relu'))
model.add(Dropout(0.3))

# Third LSTM layer with 80 units
model.add(LSTM(units=80, return_sequences=True, activation='relu'))
model.add(Dropout(0.4))

# Fourth LSTM layer with 120 units, no return_sequences=True as we don't need sequences at the end
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

# Output layer with 1 unit for regression task
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')  # Compiling the model
model.fit(x_train, y_train, epochs=20, batch_size=32)        # Training the model
model.save('spm.h5')

# Prepare test data
data_testing = df["Close"][int(len(df) * 0.7):]  # Using remaining 30% of data for testing
final_df = data_training.append(data_testing, ignore_index=True)
input_data = scaler.transform(np.array(final_df).reshape(-1, 1))

x_test = []
y_test = []
# Creating sequences of 100 days for testing
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i, 0])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape for LSTM input
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predictions
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel("Price")
plt.legend()
plt.show()
