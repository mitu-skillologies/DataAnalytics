# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Dataset and sep IV & DV
dataset = pd.read_csv('DataKit/Salary_Data.csv')
print("Dataset Selected: \n",dataset.head())
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
print("Splitted into IV and DV:\n IV", X, "\n DV", y)

# Splitting your DS into Training & Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
print("\n Splitting into Training & Testing:\n X_train & X_test",
      X_train, X_test, "\n Y_train, Y_test", y_train, y_test)

# Fitting Training Dataset to Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train[0:], y_train[0:])
print("Training the Machine")
# Prediction of Test Set results
y_pred = regressor.predict(X_test)
print("Predicted values are:\n X_test\t\ty_pred\t\t\ty_test")
for i in range(len(X_test)):
    print(X_test[i],'\t',y_pred[i],'\t',y_test[i])

# Visualising the Training set
plt.scatter(X_train, y_train, color='blue', label='REAL')
plt.plot(X_train, regressor.predict(X_train), color='green', label='BEST FIT')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# for Test set
plt.scatter(X_test, y_test, color='blue', label='REAL')
plt.plot(X_train, regressor.predict(X_train), color='green', label='BEST FIT')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()