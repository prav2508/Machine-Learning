import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

   
data = pd.read_csv('E:\Workspace\Machine_Learning\Machine Learning concepts\Part 1 - Data Preprocessing\Data.csv')



x = data.iloc[:, :-1].values
y = data.iloc[:, 3].values

#handling missing data

imputer = SimpleImputer(np.nan, strategy='mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


#categorical data 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
x = ct.fit_transform(x)

le = LabelEncoder()
y = le.fit_transform(y)


#splitting training and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#feature scaling

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
 

 
