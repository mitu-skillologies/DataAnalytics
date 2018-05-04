# importing the library
import numpy as np

# Creating a 1D Array in np
arr1 = np.array([1,2,3,4])

# Creating a 2D array
arr2 = np.array([[1,2], ['a','b']])

# Creating array with Custom Dtypes
arr3 = np.array([1,2,3], dtype=float)
arr3 = np.array([1,2,3], dtype=complex)
arr3 = np.array([1,2,3], dtype=int)
arr3 = np.array([1,-2,3], dtype=np.int8)
arr3 = np.array([1,2,-3], dtype=np.int64)
arr3 = np.array([1,2,-3], dtype=np.uint8)
arr3 = np.array([7,4,np.nan], dtype=np.int32)

# Array Attributes
arr4 = np.array([[1,2,3], [2,4,6], [1,3,5]], dtype=np.int16)

# Creating unstructured Dtype
my_dtype= ([('age', np.int16)])
arr5 = np.array([(10,), (24,), (16,)], dtype=my_dtype)
arr5['age']

# Creating a Structured Dtype
my_dtype1 = ([('Product', 'S20'), ('Price', np.int), ('Fats', np.float16)])
arr6 = np.array([('Lays', 10, 23.16), ('Pringles', 25, 28.96), ('Bingo', 10, 16.25)], dtype=my_dtype1)

# Attributes
print(arr6.dtype)
print(arr4.ndim)
array = np.array([1,2,3,5,4,7,9,2,8,5,4,7])
print(array.reshape(5,2).ndim)
array.reshape(2,5)
array_3d = array.reshape(2,3,2)

import pandas as pd
pnl = pd.Panel(array_3d)

array_3d.itemsize
x = np.array([1,2,3,4], dtype=np.int64)
x.itemsize
x.flags

desc = np.array([('C_CONTIGUOUS (C)','The data is in a single, C-style contiguous segment',
           'F_CONTIGUOUS (F)', 'The data is in a single, Fortran-style contiguous segment',
           'OWNDATA (O)', 'The array owns the memory it uses or borrows it from another object',
           'WRITEABLE (W)','The data area can be written to. Setting this to False locks the '
                           'data, making it read-only',
           'ALIGNED (A)', 'The data and all elements are aligned appropriately for the hardware',
           'UPDATEIFCOPY (U)', 'This array is a copy of some other array. When this array is unallocated, the base '
                               'array will be updated with the contents of this array'
           )], dtype=str)
desc = desc.reshape(6,2)

# Array Creation Routines

for i in range(10):
    print(i)

for i in range(5,10):
    print(i)

for i in range(1, 10, 2):
    print(i)

for i in np.arange(2, 10):
    print(i)

for i in np.arange(start=1, stop=10, step=0.5):
    print(i)

for i in np.arange(start=1, stop=10, step=0.12089):
    print(i)

# Indexing i NUMPY
nos = np.arange(10, 20)
nos[3:8]

twoD = np.arange(10, 60)
twoD = twoD.reshape(10,5)
twoD[[0,1,2], [0,1,2]]

try:
    for i in range(len(twoD)):
        print(twoD[i, i])
except IndexError:
    print('Exiting')

c = slice(2,9,2)
print(array[c])
print(array[2:8:2])

# Broadcasting
odd = np.array([1,3,5,7,9,11])
even = np.array([2,4,6,8,10,12])
mul = odd * even
new_2d = twoD[[0,1,2,3], 0:5] * even[0:5]

# Iterating over array
nos1 = np.arange(0,60,5)
nos2 = nos1.reshape(3,4)

name = 'this is a name'
for c in name:
    print(c)

li = [1, 2, 'a', 3.4j, 22.14]
for i in li:
    print(i)

for j in np.nditer(nos1):
    print(j)

# Trigonometry
# Sine
angles = np.array([0,30,45,90,180,360])
print(np.sin(angles*np.pi/180))

print(np.arcsin(0.499771))
inv = np.arcsin(0.500799)
deg = np.degrees(inv)

# ROund off
unrounded = np.arange(10, 20, 0.35882)
np.around(unrounded, decimals=-1)

np.ceil(unrounded)
np.floor(unrounded)

# Arithmetics
mat1 = np.array([1,4,8])
mat2 = np.array([[1,1,1], [2,2,2], [3,3,3]])
np.add(mat1,mat2)
np.multiply(mat1, mat2)
np.subtract(mat1,mat2)
np.power(mat1,3)

# Statistical functions
np.ptp(mat2, axis=1)
np.median(mat1)
np.mean(mat1)
np.std(mat1)
np.var(mat2)

# Sorting Methods
# a = Array
# axis = Row / Column
# Kind = types of sorting algo
# order = fields of order
mat1 = np.array([7,3,5,1,3,8,6])
np.sort(mat1)

mat2 = np.array([[4,1,9], [3,6,2], [4,0,7]])
np.sort(mat2, axis=1)

np.sort(mat1, kind='heapsort')

# The Where Clause

arr = np.arange(9).reshape(3,3)
op = np.where(arr%3==0)
bl = np.extract(arr%2==0, arr)

np.ones((3,3), dtype=int)
np.zeros((2,3), dtype=int)