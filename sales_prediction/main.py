# -*- coding: utf-8 -*-
"""Encryptix

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BH05_7bZ7qA0WXnHH1Pa9I1rycnTaPJX

# Sales Prediction
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('advertising.csv')
print(df.head())

# missing values
print(df.isnull().sum())
# Summary
print(df.describe())

X = df[['TV', 'Radio', 'Newspaper']]

# Target variable: Sales
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Linear Regression model
model = LinearRegression()
# model training
model.fit(X_train, y_train)


# predictions on the test set
y_pred = model.predict(X_test)

# R² score (coefficient of determination)
r2 = r2_score(y_test, y_pred)
print(f'R² score: {r2:.2f}')

#  Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

new_data = pd.DataFrame({'TV': [150.0], 'Radio': [22.0], 'Newspaper': [12.0]})

# Predict sales
predicted_sales = model.predict(new_data)
print(f'Predicted Sales: {predicted_sales[0]:.2f} units')