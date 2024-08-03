import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

root_folder_path = 'C:/Users/camer/Documents/Masters Thesis/Data/Training data'
specific_folder = '/1 sec batch full trading day'
folder_path = root_folder_path + specific_folder

data_sets = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        data_set = pd.read_csv(file_path)
        data_sets.append(data_set)

large_csv = pd.concat(data_sets, ignore_index=True)
large_csv.to_csv(f'{folder_path}/all_data.csv', index=False)

training_data = large_csv

def prepare_data(training_data):

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

    return X_train_scaled, X_test_scaled, y_train, y_test

def create_DFBA_model(X_train_scaled, X_test_scaled, y_train, y_test):

    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(timesteps, features)))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.2, verbose=1)
    test_loss = model.evaluate(X_test_scaled, y_test, verbose=1)

    return history, test_loss

if __name__ == "__main__":

    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data()

    history, test_loss = create_DFBA_model(X_train_scaled, X_test_scaled, y_train, y_test)

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

