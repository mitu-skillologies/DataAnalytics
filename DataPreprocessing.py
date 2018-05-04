# importing Libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('DataKit/Data.csv')

# Seperating IV and DV
X = dataset.iloc[0:, :-1]   # IV
y = dataset.iloc[0:, -1]    # DV

# Handling Missing data
# pd.isnull(X)
# X['Age'] = X['Age'].fillna(np.mean(X['Age']))
# X['Salary'] = X['Salary'].fillna(np.mean(X['Salary']))
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean')
imp1 = imp.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imp.fit_transform(X.iloc[:, 1:3])

# Encoding the Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
X.iloc[:, 0] = le_x.fit_transform(X.iloc[:, 0])

ohe_X = OneHotEncoder(categorical_features=[0])
X = ohe_X.fit_transform(X).toarray()

y = le_x.fit_transform(y)

# Splitting Dataset into Traing and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

np.var(X[:, [3, 4]])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)