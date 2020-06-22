# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
#importing data file
data = pd.read_csv('Part 2 - Regression\Part 2 - Multiple Linear Regression\Startups.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

#categorical data 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#splitting training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#No need to perform feature scaling in multiple linear regression since the variable constant in hypothesis will compensate the difference in numbers

#print("{}\n{}".format(x_train,y_train))

#modelling
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predicting
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#print("expected:{}\npredicted:{}".format(y_pred,y_test))

#suppose if prediction required on single request
print(regressor.predict([[1, 0, 0, 160000, 120000, 300000]])) #predict takes[[...]] since predict method expects a 2D array.

#Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)

#Profit=86.6×Dummy State 1−873×Dummy State 2+786×Dummy State 3−0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53