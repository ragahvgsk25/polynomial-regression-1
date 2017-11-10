# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 23:11:34 2017

@author: Raghav
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1]
y = dataset.iloc[:, 2]

#we can skip splitting the dataset and scaling the dataset

#Fitting the linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin1_reg = LinearRegression()
lin1_reg.fit(X, y)

#Fitting the polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly , y)
lin2_reg = LinearRegression()
lin2_reg.fit(X_poly , y)

#Viewing the linear regression
plt.scatter(X,y, color = 'blue')
plt.plot ( X, lin1_reg.predict(X), color = 'red')
plt.title('Linear regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Viewing the polynomial regression
plt.scatter(X,y, color = 'blue')
plt.plot ( X, lin2_reg.predict(poly_reg.fit_transform(X)), color = 'red')
plt.title('Polynomial regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

 pd.to_numeric(X)

#Viewing the polynomial regression for higher resolution and smoother curve
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color = 'blue')
plt.plot ( X_grid, lin2_reg.predict(poly_reg.fit_transform(X_grid)), color = 'red')
plt.title('Polynomial regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


