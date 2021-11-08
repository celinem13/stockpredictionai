import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

dataset_train = pd.read_csv("C:/Users/mc_ry/PycharmProjects/pythonProject3/csv/google.csv")
dataset_train.head()

training_set = dataset_train.iloc[:, 1:2].values
print(training_set)
print(training_set.shape)

my_scaled = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = my_scaled.fit_transform(training_set)

print(scaled_training_set)

x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(scaled_training_set[i - 60:i, 0])
    y_train.append((scaled_training_set[i, 0]))
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100, batch_size=32)

dataset_test = pd.read_csv("C:/Users/mc_ry/PycharmProjects/pythonProject3/csv/google.csv")
actual_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

x_test = []
for i in range(60, 80):
    x_test.append(inputs[i - 60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = model.predict(x_test)
predicted_stock_price = my_scaled.inverse_transform(predicted_stock_price)

plt.plot(actual_stock_price, color='red', label='Actual Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
