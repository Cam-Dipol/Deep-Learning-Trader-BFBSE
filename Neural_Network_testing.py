
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import joblib

filepath = 'C:/Users/camer/Documents/Masters Thesis/Data/Training data/1 sec batch/'

data1 = pd.read_csv(f'{filepath}Full_proportion_test_1secbatch_10schedulerep.csv')
data2 = pd.read_csv(f'{filepath}Full_proportion_test_1secbatch_10schedulerepv2.csv')
data3 = pd.read_csv(f'{filepath}Full_proportion_test_1secbatch_10schedulerepv3.csv')
data4 = pd.read_csv(f'{filepath}Full_proportion_test_1secbatch_10schedulerepv4.csv')
data5 = pd.read_csv(f'{filepath}Full_proportion_test_1secbatch_10schedulerepv5.csv')

training_data = pd.concat([data1,data2,data3,data4,data5], ignore_index= True )


X = training_data.drop(columns=['final_trade_price', 'time_of_trade'])
y = training_data['final_trade_price']

X = X.values
y = y.values

X = X.reshape((X.shape[0], 1, X.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_2d = X_train.reshape((X_train.shape[0], X_train.shape[2]))
X_test_2d = X_test.reshape((X_test.shape[0], X_test.shape[2]))

X_train_scaled = scaler.fit_transform(X_train_2d)
X_test_scaled = scaler.transform(X_test_2d)

scaler_filename = 'scaler.joblib'
joblib.dump(scaler, scaler_filename)

X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.2, verbose=1)
test_loss = model.evaluate(X_test_scaled, y_test, verbose=1)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label=f'Test Loss: {test_loss:.4f}')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

model.save('Neural_network_models/simple_test_model.keras')