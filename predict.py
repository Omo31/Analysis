#This project predicts the stock market

#libraries needed to predict the dataset
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#Reading and cleaning the dataset
sphist = pd.read_csv('sphist.csv')

sphist['Date'] = pd.to_datetime(sphist['Date'])
sphist['after'] = sphist['Date'] > datetime(year=2015, month=4, day=1)
sphist = sphist.sort_values(by='Date', ascending=True)

sphist['day_5'] = sphist['Close'].rolling(5).mean().shift(1)
sphist['day_30'] = sphist['Close'].rolling(30).mean().shift(1)
sphist['day_365'] =sphist['Close'].rolling(365).mean().shift(1)

sphist['std_5'] = sphist['Close'].rolling(5).std().shift(1)
sphist['std_365'] = sphist['Close'].rolling(365).std().shift(1)

sphist['day_5_volume'] = sphist['Volume'].rolling(5).mean().shift(1)
sphist['day_365_volume'] = sphist['Volume'].rolling(365).mean().shift(1)
sphist['5_volume_std'] = sphist['day_5_volume'].rolling(5).std().shift(1)


#Cleaning the dataset and making it suitable for predictions
sphist = sphist[sphist['Date'] > datetime(year=1951, month=1, day=3)]
sphist = sphist.dropna(axis=0) #This code is used to drop missing rows

train = sphist[sphist['Date'] < datetime(year=2013, month=1, day=1)]
test = sphist[sphist['Date'] >= datetime(year=2013, month=1, day=1)]


#Training the model
lr = LinearRegression()
cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date', 'after']
out_cols = train[cols]

features = train.drop(cols, axis=1)
col_2 = features.columns
target = test[col_2]

#fitting the model so as to make predictions
lr.fit(features, train['Close'])
predictions = lr.predict(target)
mae = mean_absolute_error(test['Close'], predictions)
mse = mean_squared_error(test['Close'], predictions)

# print('mae', mae, '\n')
# print('mse', mse,)

#This code is used to make predictions one day ahead
train2 = sphist.iloc[:-1]
test2 = sphist.iloc[-1:]

lr = LinearRegression()
coln = ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date', 'after']

out_coln = train2[coln]
feature = train2.drop(coln, axis=1)

col = feature.columns
target = test2[col]

lr.fit(feature, train2['Close'])
prediction = lr.predict(target)
mae2 = mean_absolute_error(test2['Close'], prediction)
mse2 = mean_squared_error(test2['Close'], prediction)

#The final results of both prediction. 
#The result include the `absolute mean error` and `mean squared error`
print('mae', mae, '\n')
print('mse', mse,)
print('\n', '\n')
print('mae2', mae2, '\n')
print('mse2', mse2,)