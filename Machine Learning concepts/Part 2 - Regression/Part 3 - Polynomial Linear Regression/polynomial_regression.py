# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Part 2 - Regression\Part 3 - Polynomial Linear Regression\Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
poly_feat = PolynomialFeatures(degree=4) #Higher the degree more likely the fit will be
x_poly = poly_feat.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
# Visualising the Linear Regression results
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('position')
plt.ylabel('salary')
#plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg2.predict(x_poly), color='blue')
plt.title('polynomial Regression')
plt.xlabel('position')
plt.ylabel('salary')
#plt.show()
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_axis = np.arange(min(X),max(X),0.1)
x_axis = x_axis.reshape((len(x_axis),1))
plt.scatter(X,y, color='red')
plt.plot(x_axis,lin_reg2.predict(poly_feat.fit_transform(x_axis)), color='blue')
plt.title('polynomial Regression smooth')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_reg2.predict(poly_feat.fit_transform([[6.5]])))