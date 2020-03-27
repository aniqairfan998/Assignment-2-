# Aniqa Irfan (Batch 3)
# Machine Learning Assignment 2
#Polynomial Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('headbrain.csv')
X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, 3:4].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = '#00CED1')
plt.plot(X, lin_reg.predict(X), color = 'grey')
plt.title('Brain weight from Head Size (Linear Regression)',fontweight='bold', fontname='Lucida Handwriting', fontsize=10 , color = 'grey')
plt.xlabel('Head Size')
plt.ylabel('Brain Weight')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = '#00CED1')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'grey')
plt.title('Brain weight from Head Size (Polynomial Regression)',fontweight='bold', fontname='Lucida Handwriting', fontsize=10 , color = 'grey' )
plt.xlabel('Head Size',fontweight='bold', fontname='Lucida Handwriting', fontsize=14 , color = '#00CED1')
plt.ylabel('Brain Weight',fontweight='bold', fontname='Lucida Handwriting', fontsize=14 , color = '#00CED1')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = '#00CED1')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'grey')
plt.title('Brain weight from Head Size (Polynomial Regression)',fontweight='bold', fontname='Lucida Handwriting', fontsize=10 , color = 'grey')
plt.xlabel('Head Size',fontweight='bold', fontname='Lucida Handwriting', fontsize=14 , color = '#00CED1')
plt.ylabel('Brain Weight ',fontweight='bold', fontname='Lucida Handwriting', fontsize=14 , color = '#00CED1')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

