# Importing Libraries
import pandas as pd
import numpy as np

# SERIES:   1 D Array format
# DATAFRAME: 2D Array format
# PANEL:       3D Array

#  ----------- PANDAS: SERIES
array = [1, 2, 3, 4]
series = np.array(array)
pd_ser1 = pd.Series(series)

new_array = ['fruit1', 'fruit2', 'fruit3', 'fruit4', 'fruit5', 'fruit6']
array = np.array(['apple','mango','strawberry','kiwi','banana','chico'])
ser2 = pd.Series(data=array, index=new_array[0:len(array)])

# Alphabet
data_dict = {'a':100, 'b':101, 'c':102, 'd':103}
ser3 = pd.Series(data_dict)

# Scalar Value Filling
ser4 = pd.Series(2, index=[1,2,3,4,5])

chr(65)
alphabet = []
asci = []

for i in range(97,123):
    alphabet.append(chr(i))
    asci.append(i)
ser5 = pd.Series(data=alphabet, index=asci)

# Slicing in Series
ser6 = pd.Series(data=asci, index=alphabet)
ser6[['a','b','z']]

# ----------------PANDAS: DATAFRAME

# Initialization
df1 = pd.DataFrame()

# List to DF
pylist = ['iron man', 'thor', 'hulk', 'spiderman', 'Dr. Strange', 'Wanda', 'Scarlet']
df2 = pd.DataFrame(pylist)

# Series to DF
df3 = pd.DataFrame(ser2)

# Dictionary to DF
dict1 = {'Tejas':['R&D', 12203, '12-03'],
         'Aniket': ['R&D', 12204, '09-04']}
df4 = pd.DataFrame(data=dict1, index=['Position', 'ID', 'DOB'])

data = [['iron man', 90], ['thor', 95], ['groot', 70], ['wanda', 96]]
df_lol = pd.DataFrame(data, columns=['Avenger', 'Performance'])

# NDArray to DF
num_array = np.ndarray(shape=(2,2), dtype=float)
df_ndarr = pd.DataFrame(num_array)

# list of Dictionary to DF
user_details = [
    {'Tejas':12203, 'Tushar':12202, 'Rashmi':12201},
    {'Tushar': '5th Dec', 'Rashmi': '6th Dec'}
]
df_lod = pd.DataFrame(user_details, index=['ID', 'DOJ'])

# Dataframe and its functions

# Create a new Dataframe
ser1 = pd.Series([1, 2, 3], index=['Interns', 'Associates', 'Projects'])
ser2 = pd.Series([3, 4, 5, 6], index=['Interns', 'Associates', 'Projects', 'Trainings'])

Company = {'Tejas': ser1, 'Aniket': ser2}

df = pd.DataFrame(Company)

print(df['Aniket'])

# Adding a new Column with Data in DF
df['Deep'] = pd.Series([6, 2, 1, 5], index=['Interns', 'Associates', 'Projects', 'Trainings'])

# Addition of 2 Columns
df['Raj'] = df['Aniket'] + df['Deep']

# Deleting a Column
del df['Deep']
df.pop('Tejas')

# Deleting a Row
df = df.drop('Associates')

# Renaming Columns and Indexes
df = df.rename(index={'Interns':0, 'Projects':1})
df.rename(columns={})

# Traversing Dataframe
print(df.iloc[0:, 0:])
print(df.iloc[0:, [0, 2]])

small_df = df.iloc[0:, [0, 2]]
one_col = df.iloc[0:, 1]

# Stats from DF
print(df.dtypes)
print(df.info())
print(df.shape)
print(df.size)
print(df['Aniket'].sum())

# Precise Printing of DataFrame
print(df.head(2))
print(df.tail(2))

# Handling Null
df = df.fillna(5)