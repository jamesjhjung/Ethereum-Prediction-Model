import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import matplotlib.pyplot as plt

df = pd.read_csv('ethereum_price_model.csv')

# train & test sets
df_train = df.iloc[:2538, 2:3].values
df_test = df.iloc[2538:, 2:3].values

# scaling
sc = MinMaxScaler(feature_range = (0, 1))
df_scaled = sc.fit_transform(df_train)

# creating data with timesteps
X_train = []
y_train = []
for i in range(60, 2538):
    X_train.append(df_scaled[i-60:i, 0])
    y_train.append(df_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# building the LSTM
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# predictions, reformatting to readable format

df_pred = df['Price']
inputs = df_pred[len(df) - len(df_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 76):
    X_test.append(inputs[i - 60:i, 0])

X_test = [x.tolist() for x in X_test]
length = max(map(len, X_train))
X_test = np.array([x + [0] * (length - len(x)) for x in X_test])

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_eth_price = regressor.predict(X_test)
predicted_eth_price = sc.inverse_transform(predicted_eth_price)

# results
plt.plot(df_test, color = 'blue', label = 'Ethereum Price')
plt.plot(predicted_eth_price, color = 'red', label = 'Predicted Ethereum Price')
plt.title('Ethereum Price Prediction')
plt.xlabel('Time')
plt.ylabel('Ethereum Price')
plt.legend()
plt.show()