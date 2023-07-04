import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#changed
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.title('Stockscope 🔍')
st.write('Welcome to your one stop point to improve you decision making regarding stocks to purchase !')
ticker= st.text_input("Enter the ticker symbol: " )

def main():
    # Prompt for the ticker symbol
    #ticker = input("Enter the ticker symbol: ")

    # Set the start and end dates
    end_date = datetime.now()
    start_date = datetime(end_date.year - 1, end_date.month, end_date.day)

    # Download the stock data
    yf.pdr_override()
    df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    df = df.reset_index()
    
    # Perform exploratory data analysis (EDA)
    eda(df, ticker)

    # Prepare the data
    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    training_size = int(len(scaled_data) * 0.70)
    test_size = len(scaled_data) - training_size
    train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :1]

    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.show()

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse_train = mean_squared_error(original_ytrain, train_predict)
    mse_test = mean_squared_error(original_ytest, test_predict)

    print("Train data MSE:", mse_train)
    print("Test data MSE:", mse_test)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Original close price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'][time_step:len(train_predict) + time_step],
                             y=train_predict.reshape(1, -1)[0].tolist(),
                             name="Train predicted close price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'][len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1],
                             y=test_predict.reshape(1, -1)[0].tolist(),
                             name="Test predicted close price"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Original close price"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'][time_step:len(train_predict) + time_step],
                             y=train_predict.reshape(1, -1)[0].tolist(),
                             name="Train predicted close price"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'][len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1],
                             y=test_predict.reshape(1, -1)[0].tolist(),
                             name="Test predicted close price"), row=2, col=1)

    fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()
    
