# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
#importing data file
dataset = pd.read_csv('Part 2 - Regression\Part 4 - Support vector Regression\Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


#reshape dependent variable to 2d array
y = y.reshape(len(y),1)

#feature scaling

sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

#modelling

svr_reg = SVR(kernel='rbf') #rbf stands from gaussion radial basis function(https://data-flair.training/blogs/svm-kernel-functions/)
svr_reg.fit(X,y)

#predicting 

y_pred = svr_reg.predict(sc_x.transform([[2]])) #we use just sc_x.transform(...) because the x feature is already used to fit, so transform will use the same configuration
y_pred = sc_y.inverse_transform(y_pred)

print(y_pred)

# Visualising the Linear Regression results
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(X),sc_y.inverse_transform(svr_reg.predict(X)), color='blue')
plt.title('Support Vector Regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_axis = np.arange(min(sc_x.inverse_transform(X)),max(sc_x.inverse_transform(X)),0.1)
x_axis = x_axis.reshape((len(x_axis),1))
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(y), color='red')
plt.plot(x_axis,sc_y.inverse_transform(svr_reg.predict(sc_x.transform(x_axis))), color='blue')
plt.title('Support Vector Regression (smooth version)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()