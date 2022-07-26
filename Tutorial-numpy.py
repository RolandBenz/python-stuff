import numpy as np

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

from math import log

import pandas as pd

import csv

"""https://numpy.org/doc/stable/index.html"""


def one_index():
    print("1.0 nympy version: \n", np.__version__)

    arr = np.array(42)
    print("1.1.1 ndarray of dimensions: ", arr.ndim)
    print(arr)

    arr = np.array([1, 2, 3, 4, 5])
    print("1.1.2 ndarray of dimensions: ", arr.ndim)
    print(arr)
    print("element 0 is: ", arr[0])
    print("element 2 + element 3 = ", arr[2] + arr[3])

    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print("1.1.3 ndarray of dimensions: ", arr.ndim)
    print(arr)
    print('element 0, 1: ', arr[0, 1])
    print('element 1, -1: ', arr[1, -1])
    print('element -1, -2: ', arr[-1, -2])

    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print("1.1.4 ndarray of dimensions: ", arr.ndim)
    print(arr)
    print("element 0, 1, 2: ", arr[0, 1, 2])

    arr = np.array([1, 2, 3, 4], ndmin=5)
    print("1.1.5 ndarray of dimensions: ", arr.ndim)
    print(arr)
    print("1.2 ndarray type: ", type(arr))


def two_slice():
    arr = np.array([1, 2, 3, 4, 5, 6, 7])
    print (arr)
    print("2.1.1 [1:5]: ", arr[1:5])
    print("2.1.2 [4:]: ", arr[4:])
    print("2.1.3 [:4]: ", arr[:4])
    print("2.1.4 [-3:-1]: ", arr[-3:-1])
    print("2.1.5 [1:5:2]: ", arr[1:5:2])
    print("2.1.5 [::2]: ", arr[::2])

    arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    print(); print (arr)
    print("2.2.1 [1, 1:4]: ", arr[1, 1:4])
    print("2.2.2 [0:2, 2]: ", arr[0:2, 2])
    print("2.2.3 [0:2, 1:4]: \n", arr[0:2, 1:4])


def three_types():
    """
    Python Datatypes:
    strings - used to represent text data, the text is given under quote marks. e.g. "ABCD"
    integer - used to represent integer numbers. e.g. -1, -2, -3
    float - used to represent real numbers. e.g. 1.2, 42.42
    boolean - used to represent True or False.
    complex - used to represent complex numbers. e.g. 1.0 + 2.0j, 1.5 + 2.5j
    Numpy Datatypes:
    i - integer (32 bit: range from -2147483648 to 2'147'483'647)
    q - long (64bit use for bigger numbers: range  from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807)
    b - boolean
    u - unsigned integer
    f - float
    c - complex float
    m - timedelta
    M - datetime
    O - object
    S - string
    U - unicode string
    V - fixed chunk of memory for other type ( void )
    """
    arr = np.array([1, 2, 3, 4])
    print("3.1.1 ndarray type: ", arr.dtype)
    arr = np.array(['apple', 'banana', 'cherry'])
    print("3.1.2 ndarray type: ", arr.dtype)

    arr = np.array([1, 2, 3, 4], dtype='S')
    print(arr)
    print("3.2.1 ndarray type: ", arr.dtype)
    arr = np.array([1, 2, 3, 4], dtype='i4')
    print(arr)
    print("3.2.2 ndarray type: ", arr.dtype)

    arr = np.array([1.1, 2.1, 3.1])
    newarr = arr.astype('i')
    print(arr)
    print(newarr)
    print("3.3.1 ndarray type before conversion: ", arr.dtype)
    print("3.3.2 ndarray type after conversion: ", newarr.dtype)
    arr = np.array([1.1, 2.1, 3.1])
    newarr = arr.astype(int)
    print(arr)
    print(newarr)
    print("3.3.3 ndarray type before conversion: ", arr.dtype)
    print("3.3.4 ndarray type after conversion: ", newarr.dtype)
    arr = np.array([1, 0, 3])
    newarr = arr.astype(bool)
    print(arr)
    print(newarr)
    print("3.3.5 ndarray type before conversion: ", arr.dtype)
    print("3.3.6 ndarray type after conversion: ", newarr.dtype)


def four_copy_vs_view():
    arr = np.array([1, 2, 3, 4, 5])
    x = arr.copy()
    print("4.1.1 array before change: ", arr)
    arr[0] = 42
    print("4.1.2 array after change of array: ", arr)
    print("4.1.3 copy of array before change array: ", x, "\n")

    arr = np.array([1, 2, 3, 4, 5])
    x = arr.view()
    print("4.2.1 array before change: ", arr)
    arr[0] = 42
    print("4.2.2 array after change of array: ", arr)
    print("4.2.3 view of array after change of array: ", x, "\n")

    arr = np.array([1, 2, 3, 4, 5])
    x = arr.view()
    print("4.3.1 array before change of view: ", arr)
    x[0] = 42
    print("4.3.2 view after change of view: ", arr)
    print("4.3.3 array after change of view: ", x, "\n")

    arr = np.array([1, 2, 3, 4, 5])
    x = arr.copy()
    y = arr.view()
    print("4.4.1 array: ", arr)
    print("4.4.2 copy owns data of array, .base returns: ", x.base)
    print("4.4.3 view does not own data of array, .base returns: ", y.base, "\n")


def five_shape():
    """arr.shape
    Integers at every index tells about the number of elements the corresponding dimension has."""

    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    print("5.1.1 array: \n", arr)
    print("5.1.2 shape: ", arr.shape)
    print("5.1.3 dimensions: ", arr.ndim, "\n")

    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print("5.2.1 array: \n", arr)
    print("5.2.2 shape: ", arr.shape)
    print("5.2.3 dimensions: ", arr.ndim, "\n")

    arr = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 12, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]]])
    print("5.3.1 array: \n", arr)
    print("5.3.2 shape: ", arr.shape)
    print("5.3.3 dimensions: ", arr.ndim, "\n")

    arr = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
    print("5.4.1 array: \n", arr)
    print("5.4.2 shape: ", arr.shape)
    print("5.4.3 dimensions: ", arr.ndim, "\n")

    arr = np.array([1, 2, 3, 4], ndmin=5)
    print("5.5.1 array: \n", arr)
    print("5.5.2 shape: ", arr.shape)
    print("5.5.3 dimensions: ", arr.ndim, "\n")


def six_reshape():
    """Note: There are a lot of functions for changing the shapes of arrays in numpy
    flatten, ravel and also for rearranging the elements rot90, flip, fliplr, flipud etc.
    These fall under Intermediate to Advanced section of numpy."""
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    newarr = arr.reshape(2, 4)
    print("6.1.1 array: \n", arr)
    print("6.1.2 reshaped (4,3) array: \n", newarr)
    print("6.1.3 reshaped arrays are views not copies: ", newarr.base, "\n")

    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    newarr = arr.reshape(4, 3)
    print("6.2.1 array: \n", arr)
    print("6.2.2 reshaped (4,3) array: \n", newarr, "\n")

    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    newarr = arr.reshape(2, 3, 2)
    print("6.3.1 array: \n", arr)
    print("6.3.2 reshaped (2, 3, 2) array: \n", newarr, "\n")

    arr = np.array([[1, 2, 3], [4, 5, 6]])
    newarr = arr.reshape(-1)
    print("6.4.1 array: \n", arr)
    print("6.4.2 reshaped flattened (-1) array: \n", newarr, "\n")


def seven_iterate():
    """We can use op_dtypes argument and pass it the expected datatype
    to change the datatype of elements while iterating.
    NumPy does not change the data type of the element in-place (where the element is in array)
    so it needs some other space to perform this action,
    that extra space is called buffer, and in order to enable it
    in nditer() we pass flags=['buffered']."""

    arr = np.array([1, 2, 3])
    print("7.1.1 array: \n", arr)
    for x in arr:
        print("7.1.2 print elements: ", x)
    print()

    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print("7.2.1 array: \n", arr)
    for x in arr:
        print("7.2.2 print elements: ", x)
    print()
    for x in arr:
        for y in x:
            print("7.2.3 print elements: ", y)
    print()

    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print("7.3.1 array: \n", arr)
    for x in arr:
        print("7.3.2 print elements: \n", x)
    print()
    for x in arr:
        for y in x:
            for z in y:
                print("7.3.3 print elements: ", z)
    print()

    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print("7.4.1 array: \n", arr)
    for x in np.nditer(arr):
        print("7.4.2 one for loop with nditer, print elements: ", x)
    print()

    arr = np.array([1, 2, 3])
    print("7.5.1 array: \n", arr)
    print("7.5.2 ndarray type: ", arr.dtype)
    for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
        print("7.5.3 one for loop with nditer, change dtypes, print elements: ", x)
        print("7.5.4 ndarray type: ", x.dtype)
    print()

    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    print("7.6.1 array: \n", arr)
    for x in np.nditer(arr[:, ::2]):
        print("7.6.2 one for loop with nditer(arr[:, ::2]), print elements: ", x)

    arr = np.array([1, 2, 3])
    print("7.7.1 array: \n", arr)
    for idx, x in np.ndenumerate(arr):
        print("7.7.2 one for loop with ndenumerate, print indices and elements: ", idx, x)
    print()

    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    print("7.8.1 array: \n", arr)
    for idx, x in np.ndenumerate(arr):
        print("7.8.2 one for loop with ndenumerate, print indices and elements: ", idx, x)
    print()


def eight_join():
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    print("8.1.1 arrays: \n", arr1, arr2)
    print()

    arr = np.concatenate((arr1, arr2))
    print("8.1.2 concatenated arrays: \n", arr)
    arr = np.stack((arr1, arr2), axis=1)
    print("8.1.3 stacked arrays: \n", arr)
    print()

    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])
    print("8.2.1 arrays: \n", arr1, "\n", arr2)
    print()

    arr = np.concatenate((arr1, arr2), axis=0)
    print("8.2.2 concatenated arrays, axix=0: \n", arr)
    arr = np.concatenate((arr1, arr2), axis=1)
    print("8.2.3 concatenated arrays, axix=1: \n", arr)
    print()

    arr = np.stack((arr1, arr2), axis=0)
    print("8.2.4 stacked arrays, axis=0: \n", arr)
    arr = np.stack((arr1, arr2), axis=1)
    print("8.2.5 stacked arrays, axis=1: \n", arr)
    print()

    arr = np.hstack((arr1, arr2))
    print("8.2.6 hstacked arrays: \n", arr)
    arr = np.vstack((arr1, arr2))
    print("8.2.7 vstacked arrays: \n", arr)
    arr = np.dstack((arr1, arr2))
    print("8.2.7 dstacked arrays: \n", arr)
    print()


def nine_split():
    """There are some differences in split and array_split"""
    arr = np.array([1, 2, 3, 4, 5, 6])
    print(arr)
    newarr = np.array_split(arr, 3)
    print("9.1.1 array_split 3 array: \n", newarr)
    newarr = np.split(arr, 3)
    print("9.1.2 split 3 array: \n", newarr)
    print()

    arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    print(arr)
    newarr = np.array_split(arr, 3)
    print("9.2.1 array_split 3 array: \n", newarr)
    print(newarr)

    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
    print(arr)
    newarr = np.array_split(arr, 3, axis=1)
    print("9.2.1 array_split 3, axis=1, array: \n", newarr)


def ten_search():
    arr = np.array([1, 2, 3, 4, 5, 4, 4])
    print(arr)
    x = np.where(arr == 4)
    print("10.1.1 print indexes where element in array == 4 : \n", x)
    print()
    x = np.where(arr % 2 == 0)
    print("10.1.2 print indexes where element in array == even : \n", x)
    print()
    x = np.where(arr % 2 == 1)
    print("10.1.3 print indexes where element in array == uneven : \n", x)
    print()


def eleven_sort():
    arr = np.array([3, 2, 0, 1])
    print(arr)
    x= np.sort(arr)
    print("11.1.1 print sort: ", x)
    print("11.1.2 print sort: ", x.base)
    x[1]=5
    print("11.1.3 print sort: ", x)
    print("11.1.4 print sort: ", arr)

    arr = np.array(['banana', 'cherry', 'apple'])
    print(arr)
    x=np.sort(arr)
    print("11.1.1 print sort: ", x)


def twelve_filter():
    arr = np.array([41, 42, 43, 44])
    print(arr)
    x = [True, False, True, False]
    newarr = arr[x]
    print("12.1.1 print filtered array: ", newarr)
    print()

    newarr = []
    print("12.2.1 newarr: ", newarr, type(newarr))
    for element in arr:
        if element > 42:
            newarr = np.append(newarr, element)
    print("12.2.2 newarr: ", newarr, type(newarr), newarr.dtype)
    print()

    filter_arr = []
    for element in arr:
        if element > 42:
            filter_arr.append(True)
        else:
            filter_arr.append(False)
    newarr = arr[filter_arr]
    print("12.3.1 newarr: ", filter_arr)
    print("12.3.2 newarr: ", newarr)
    print()

    filter_arr = arr > 42
    newarr = arr[filter_arr]
    print("12.4.1 newarr: ", filter_arr)
    print("12.4.2 newarr: ", newarr)
    print()

    filter_arr = arr % 2 == 0
    newarr = arr[filter_arr]
    print("12.5.1 newarr: ", filter_arr)
    print("12.5.2 newarr: ", newarr)
    print()


def thirteen_random():
    x = random.randint(100)
    print("13.1.1 random from 0 to 100: ", x)
    x = random.rand()
    print("13.1.2 random from 0 to 1: ", x)
    x = random.randint(100, size=(3, 5))
    print("13.1.3 random 2d array from 0 to 100: \n", x)
    x = random.choice([3, 5, 7, 9])
    print("13.1.4 random choose on in [3,5,7,9]: ", x)
    x = random.choice([3, 5, 7, 9], size=(3, 5))
    print("13.1.5 random choose on in [3,5,7,9]: \n", x)

    x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))
    print("13.2.1 random choose on in [3,5,7,9] with p=[0.1, 0.3, 0.6, 0.0]: \n", x)

    sns.distplot([0, 1, 2, 2, 2, 3, 4, 5], axlabel="13.3.1 histogram and distribution")
    plt.show()
    sns.distplot(random.normal(size=1000), hist=True, axlabel='13.3.2 normal distribution')
    plt.show()
    sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False, axlabel="13.3.3 binomial distribution")
    plt.show()
    sns.distplot(random.poisson(lam=2, size=1000), hist=True, kde=False, axlabel="13.3.4 poisson distribution")
    plt.show()
    sns.distplot(random.uniform(size=1000), hist=True, axlabel="13.3.5 uniform distribution")
    plt.show()


def fourteen_ufunc():
    """ufuncs are used to implement vectorization in NumPy
    which is way faster than iterating over elements."""
    x = [1, 2, 3, 4]
    y = [4, 5, 6, 7]
    z = []
    for i, j in zip(x, y):
        z.append(i + j)
    print("14.1.1 add without ufunc: ", z)

    z = np.add(x, y)
    print("14.2.1 add with ufunc: ", z)
    print("14.2.2 check if type ufunc: ", type(np.add))

    myadd = np.frompyfunc(fourteen_myadd, 2, 1)
    print("14.3.1 add with ufunc: ", myadd([1, 2, 3, 4], [5, 6, 7, 8]))

    arr1 = np.array([10, 11, 12, 13, 14, 15])
    arr2 = np.array([20, 21, 22, 23, 24, 25])
    newarr = np.add(arr1, arr2)
    print("14.4.1 add with ufunc: ", newarr)
    newarr = np.subtract(arr1, arr2)
    print("14.4.2 subtract with ufunc: ", newarr)
    newarr = np.multiply(arr1, arr2)
    print("14.4.3 multiply with ufunc: ", newarr)
    newarr = np.divide(arr1, arr2)
    print("14.4.4 divide with ufunc: ", newarr)
    arr1 = np.array([10, 20, 30, 40, 50, 60], dtype='q')
    arr2 = np.array([3, 5, 6, 8, 2, 33], dtype='q')
    newarr = np.power(arr1, arr2, dtype='q')
    print("14.4.5 power with ufunc use big enough datatype: ", newarr)
    newarr = np.mod(arr1, arr2)
    print("14.4.6 mod with ufunc: ", newarr)
    arr = np.array([-1, -2, 1, 2, 3, -4])
    newarr = np.absolute(arr)
    print("14.4.7 absolute with ufunc: ", newarr)
    print()

    arr = np.arange(1, 10)
    print("14.5.1 log2: ", np.log2(arr))
    print("14.5.2 log10: ", np.log10(arr))
    print("14.5.3 log: ", np.log(arr))
    nplog = np.frompyfunc(log, 2, 1)
    print("14.5.4 math.log(100,15): ", nplog(100, 15))
    print()

    arr1 = np.array([1, 2, 3])
    arr2 = np.array([1, 2, 3])
    x = np.sum([arr1, arr2])
    print("14.6.1 sum: ", x)
    x = np.prod([arr1, arr2])
    print("14.6.2 prod: ", x)
    print()

    x=np.arange(-3, 3, 0.1, dtype=float)
    y = np.sin(x)
    print(y)
    plt.plot(x,y)
    plt.show()


def fourteen_myadd(x, y):
    return x+y


def fifteen_withImportedData():
    # read the csv file
    # (put 'r' before the path string to address any special characters in the path, such as '\').
    # Don't forget to put the file name at the end of the path + ".csv"

    # using numpy
    path1 = r'C:\Users\41792\Documents\4) Python-Scripts\Pack_Numpy\data-numbers.csv'
    f = open(path1)
    header = f.readline()
    print("header: ", header)

    mydata = np.loadtxt(path1, delimiter=',', usecols=(0, 1, 2), unpack=True, dtype=int, skiprows=(1))
    print("np.loadtxt")
    print("type: ", type(mydata))
    print("shape: ", mydata.shape)
    print("dimensions: ", mydata.ndim, "\n")
    print(mydata)
    print("slices: \n", mydata[[0, 1], 2:4])
    print("slices: \n", mydata[[0,2]])
    print()

    mydata = np.genfromtxt(path1, delimiter=',', usecols=(1, 2), unpack=True, dtype=None, skip_header=True)
    print("np.genfromtxt")
    print("type: ", type(mydata))
    print("shape: ", mydata.shape)
    print("dimensions: ", mydata.ndim, "\n")
    print(mydata)
    print()

    mydata = np.recfromcsv(path1, delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='',
                           replace_space=' ')
    print("np.recfromcsv")
    print("type: ", type(mydata))
    print("shape: ", mydata.shape)
    print("dimensions: ", mydata.ndim, "\n")
    print(mydata)
    print("slices: \n", mydata[1:5]['Offer'])
    print("slices: \n", mydata[:]['Price'])
    print("slices: \n", mydata[[0, 2]])
    print()

    # using csv
    with open(path1, 'r') as f:
        mydata_ = list(csv.reader(f, delimiter=";"))
    print("csv.reader")
    print(mydata_)
    slice = []
    for x in mydata_[1:]:
        slice.append(x[0].split(',')[1:3])
        print("type x: ", type(x))
        print("type x[0]: ", type(x[0]))
    print("slices: \n", slice)

    # using pandas
    path1 = r'C:\Users\41792\Documents\4) Python-Scripts\Pack_Numpy\data-numbers.csv'
    df1 = pd.read_csv(path1)
    print("type: ", type(df1))
    print("shape: ", df1.shape)
    print("dimensions: ", df1.ndim, "\n")
    print(df1)
    print("slices: \n", df1.iloc[2:4,[0,1]])
    print("slices: \n", df1.loc[2:8, ['Offer', 'Price']])

    path2 = r'C:\Users\41792\Documents\4) Python-Scripts\Pack_Numpy\data.csv'
    df2_=pd.read_csv(path2)
    df2 = pd.DataFrame(df2_, columns=['Name', 'Country'])
    print(df2)

    path3 = r'C:\Users\41792\Documents\4) Python-Scripts\Pack_Numpy\stats.csv'
    df3 = pd.read_csv(path3)
    print(df3)
    fifteen_statsWithPandas(df3)


def fifteen_statsWithPandas(df):
    # block 1 - simple stats
    mean1 = df['Salary'].mean()
    sum1 = df['Salary'].sum()
    max1 = df['Salary'].max()
    min1 = df['Salary'].min()
    count1 = df['Salary'].count()
    median1 = df['Salary'].median()
    std1 = df['Salary'].std()
    var1 = df['Salary'].var()

    # block 2 - group by
    groupby_sum1 = df.groupby(['Country']).sum()
    groupby_count1 = df.groupby(['Country']).count()

    # print block 1
    print('Mean salary: ' + str(mean1))
    print('Sum of salaries: ' + str(sum1))
    print('Max salary: ' + str(max1))
    print('Min salary: ' + str(min1))
    print('Count of salaries: ' + str(count1))
    print('Median salary: ' + str(median1))
    print('Std of salaries: ' + str(std1))
    print('Var of salaries: ' + str(var1))

    # print block 2
    print('Sum of values, grouped by the Country: \n' + str(groupby_sum1))
    print('Count of values, grouped by the Country: \n' + str(groupby_count1))


if __name__ == '__main__':
    if 0:
        one_index()
        two_slice()
        three_types()
        four_copy_vs_view()
        five_shape()
        six_reshape()
        seven_iterate()
        eight_join()
        nine_split()
        ten_search()
        eleven_sort()
        twelve_filter()
        thirteen_random()
        fourteen_ufunc()
    else:
        fifteen_withImportedData()
