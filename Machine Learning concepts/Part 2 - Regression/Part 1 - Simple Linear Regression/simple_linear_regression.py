# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Part 2 - Regression\Part 1 - Simple Linear Regression\Salary_Data.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


#splitting training and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0)


#modelling

regressor = LinearRegression()
regressor.fit(x_train,y_train) 

#predicting
y_pred =regressor.predict(x_test)
print("Test result are:- \n test data: {} \n test predicted data: {}".format(x_test,y_pred))
print("Expected result: {}\n Predicted result: {}".format(y_test,y_pred))
#vizualization
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
#plt.plot(x_test, regressor.predict(x_test),color='green')
plt.title('Salary vs Experience [training set]')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
