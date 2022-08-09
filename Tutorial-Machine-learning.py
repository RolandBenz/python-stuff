"""
each tutorial function contains a separate part of the tutorial
choose the tutorial function to run at the very end of this file
"""

"""
# imports used, but put into the separate tutorial functions, holding all code for one tutorial:
#
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import svm
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import math
import quandl
import datetime
import pickle
import random
import warnings
from math import sqrt
import cvxopt
import cvxopt.solvers
from collections import Counter
from statistics import mean
import pylab as pl
from functools import reduce
"""
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

"""
# tutorials:
# https://pythonprogramming.net/machine-learning-tutorial-python-introduction/
# https://pythonprogramming.net/flat-clustering-machine-learning-python-scikit-learn/
#
# scikit website:
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
#
# sklearn and scikit-learn reference the same package:
#   - one should use 'pip install scikit-learn' to install,
#     otherwise the 'pip uninstall sklearn' would not remove scikit-learn,
#     but only some pypi dummy file sklearn
#   - but one must use 'import sklearn'
"""
"""
# Jupyter notebook: in case you like to run part of the code there
# Jupyter notebook startup:
# 1.Terminal:
#   Tut_Machine-learning> jupyter lab
# 2. Browser (if it does not startup automatically):
#   url>  http://localhost:8888/lab?token=8cc5888a814ddf00c9546a258b27f9c9e512540fd0c1f4c1
# 3. Jupiter Lab Browser App:
#   file > new > notebook >Python 3
#   file > save notebook as... > Tut_Pandas> Tutorial-pandas.ipynb
"""


def one_k_means_clustering():
    """
    Unsupervised Machine Learning: Flat Clustering
    K-Means clustering example with Python and Scikit-learn
    Flat Clustering:
    Flat clustering is where the scientist tells the machine how many categories to cluster the data into.
    Hierarchical:
    Hierarchical clustering is where the machine is allowed to decide how many clusters to create based on its own algorithms.
    """
    from sklearn.cluster import KMeans

    """
    # generate data
    """
    x = [1, 5, 1.5, 8, 1, 9]
    y = [2, 8, 1.8, 8, 0.6, 11]
    plt.scatter(x, y)
    plt.show()
    """
    # converting our data to a NumPy array. 
    """
    data = np.array([[1, 2],
                     [5, 8],
                     [1.5, 1.8],
                     [8, 8],
                     [1, 0.6],
                     [9, 11]])
    """
    # initialize kmeans to be the KMeans algorithm (flat clustering), 
    # with the required parameter of how many clusters (n_clusters).
    """
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    """
    # found cluster labels and cluster centers
    """
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    print(centroids)
    print(labels)
    """
    # provide enough colors for found cluster labels
    """
    colors = ["g.", "r.", "c.", "y."]

    """
    # plot and visualize the machine's findings based on our data
    """
    for i in range(len(data)):
        print("coordinate:", data[i], "label:", labels[i])
        plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize=10)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    plt.show()


def two_mean_shift_clustering_2D():
    """
    Unsupervised Machine Learning: Hierarchical Clustering
    Mean Shift cluster analysis example with Python and Scikit-learn
    """
    from sklearn.cluster import MeanShift
    from sklearn.datasets import make_blobs

    """
    # we're making our example data set. 
    # We've decided to make a dataset that originates from three center-points.
    # X is the dataset, and y is the label of each data point according to the sample generation.
    """
    actual_centers = [[1, 1], [5, 5], [3, 10]]
    data, actual_labels = make_blobs(n_samples=500, centers=actual_centers, cluster_std=1)
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    print("\n2.1\n", actual_centers)
    print("\n2.2\n", actual_labels)

    """
    # Learning Algorithm:
    # Here, we can already identify ourselves the major clusters.
    # What we want is the machine to do the same thing.
    # For this, we're going to use MeanShift
    """
    ms = MeanShift()
    ms.fit(data)

    """
    # We can see the cluster centers 
    # and grab the total number of clusters by doing the following.
    """
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    n_clusters_ = len(np.unique(labels))
    print("\n2.3\nNumber of estimated clusters:", n_clusters_)
    print("\n2.4\n", cluster_centers)
    print("\n2.5\n", labels)

    """
    This is just a simple list of red, green, blue, cyan, black, yellow, and magenta multiplied by ten. 
    We should be confident that we're only going to need three colors, 
    but, with hierarchical clustering, we are allowing the machine to choose, 
    we'd like to have plenty of options. This allows for 70 clusters, so that should be good enough.
    """
    colors = 10 * ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']
    print("\n2.6\n", colors)

    """
    This code is purely for graphing only, and has nothing to do with machine learning 
    other than helping us see what is happening:
    First we're iterating through all of the sample data points, 
    plotting their coordinates, and coloring by their label # as an index value in our color list.
    Then we are calling plt.scatter to scatter plot the cluster centers.
    """
    for i in range(len(data)):
        plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize=10)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                marker="x", color='k', s=150, linewidths=5, zorder=10)
    plt.show()


def two_mean_shift_clustering_3D():
    """
    Unsupervised Machine Learning: Hierarchical Clustering
    Mean Shift cluster analysis example with Python and Scikit-learn
    """
    from sklearn.cluster import MeanShift
    from sklearn.datasets import make_blobs

    """
    # generate data
    """
    centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
    X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1.5)
    print(centers)

    """
    # learning algorithm
    """
    ms = MeanShift()
    ms.fit(X)

    """
    # found cluster labels and cluster centers
    """
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    print(cluster_centers)
    n_clusters_ = len(np.unique(labels))
    print("Number of estimated clusters:", n_clusters_)

    """
    # provide enough colors for found cluster labels
    """
    colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y', 'm']
    print(colors)
    print(labels)

    """
    # visualization
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(X)):
        ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
               marker="x", color='k', s=150, linewidths=5, zorder=10)
    plt.show()


def three_regression():
    """
    Regression:
    You could learn a lot about each algorithm to figure out which ones can thread,
    or you can visit the documentation, and look for the n_jobs parameter.
    If it has n_jobs, you have an algorithm that can be threaded for high performance.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html :
        no, no n_jobs.
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html:
        yes, n_jobs. If you put in -1 for the value, then the algorithm will use all available threads.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn import preprocessing
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    import quandl
    import math

    """
    # Get data:
    # We're able to at least start with simple stock price and volume information from Quandl. 
    # To begin, we'll start with data that grabs the stock price for Alphabet (previously Google), 
    # with the ticker of GOOGL:
    """
    df = quandl.get("WIKI/GOOGL")
    df_display(df.head(5), "3.0")

    """
    # Keep useful data:
    # Adjusted columns are adjusted for stock splits over time, 
    # which makes them more reliable for doing analysis.
    """
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

    """
    # Transform the data:
    """
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
    df_display(df.head(5), "3.1")

    """
    # Here, we define the forecasting column, then we fill any NaN data with -99999. 
    # You have a few choice here regarding how to handle missing data. 
    # You can't just pass a NaN (Not a Number) datapoint to a machine learning classifier, 
    # you have to handle for it. One popular option is to replace missing data with -99,999. 
    # With many machine learning classifiers, this will just be recognized and treated as an outlier feature. 
    # You can also just drop all feature/label sets that contain missing data, 
    # but then you're maybe leaving a lot of data out.
    """
    forecast_col = 'Adj. Close'
    df.fillna(value=-99999, inplace=True)

    """
    We've decided the features are a bunch of the current values, 
    and the label shall be the price, in the future, 
    where the future is 1% of the entire length of the dataset out.
    """
    forecast_out = int(math.ceil(0.01 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)
    #
    df_display(df.head(100), "3.2")
    df_display(df.tail(100), "3.3")
    print("\n3.4\n", forecast_out)
    print("\n3.5\n", df.shape, df.size)
    print("\n3.6\n", df.columns)
    print("\n3.7\n")
    print(df.info())

    """
    # drop any still NaN information from the dataframe
    # dataset for features X and labels y
    """
    df.dropna(inplace=True)
    X = np.array(df.drop(['label'], axis=1))
    y = np.array(df['label'])
    #
    print("\n3.8\n", X.shape, y.shape)
    print("\n3.9\n", X.ndim, y.ndim)
    print("\n3.10\n", X.size, y.size)
    print("\n3.11\n", len(X), len(y))

    """
    # Generally, you want your features in machine learning to be in a range of -1 to 1. 
    # This may do nothing, but it usually speeds up processing and can also help with accuracy. 
    # Because this range is so popularly used, it is included in the preprocessing module of Scikit-Learn.
    """
    X = preprocessing.scale(X)

    """
    # Now comes the training and testing.
    # There are many ways to do this, but, probably the best way 
    # is using the build in cross_validation provided, since this also shuffles your data for you.
    # The return here is the training set of features, 
    # testing set of features, training set of labels, and testing set of labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    """
    # There are many classifiers in general available through Scikit-Learn, 
    # and even a few specifically for regression. 
    # We'll show a couple in this example, but for now, let's use Support Vector Regression
    # with default settings
    """
    svr = False
    if svr:
        clf = svm.SVR()
        clf.fit(X_train, y_train)

    """
    # Our classifier is now trained. Now we can test it!
    """
    if svr:
        confidence = clf.score(X_test, y_test)
        print("\n3.12\n", confidence)

    """
    # LinearRegression from sklearn
    """
    regression = True
    if regression:
        # clf = LinearRegression()
        clf = LinearRegression(n_jobs=-1)  # using all available threads for calculation
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print("\n3.13\n", confidence)

    """
    SVM kernels:
    Check the documentation, you have 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. 
    Again, just like the suggestion to try the various ML algorithms that can do what you want, 
    try the kernels.
    """
    svm_kernels = False
    if svm_kernels:
        for k in ['linear', 'poly', 'rbf', 'sigmoid']:
            clf = svm.SVR(kernel=k)
            clf.fit(X_train, y_train)
            confidence = clf.score(X_test, y_test)
            print("\n3.14\n", k, confidence)


def four_regression_forecasts():
    """
    # Forecasts:
    # consider the data we're trying to forecast is not scaled like the training data was.
    # Do we just do preprocessing.scale() against the last 1% (i.e. the 35 we lost because of forecast_out)?
    # The scale method scales based on all of the known data that is fed into it.
    # Ideally, you would scale both the training, testing, AND forecast/predicting data all together.
    # Is this always possible or reasonable? No. If you can do it, you should, however.
    # In our case, right now, we can do it.
    # Our data is small enough and the processing time is low enough,
    # so we'll preprocess and scale the data all at once.
    # In many cases, you wont be able to do this.
    # Imagine if you were using gigabytes of data to train a classifier.
    # It may take days to train your classifier,
    # you wouldn't want to be doing this every...single...time you wanted to make a prediction.
    # Thus, you may need to either NOT scale anything, or you may scale the data separately.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    import pickle
    import datetime
    import quandl
    import math

    """
    # same as in three_regression()
    """
    df = quandl.get("WIKI/GOOGL")
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
    forecast_col = 'Adj. Close'
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)

    """
    We had in three_regression():
        df.dropna(inplace=True)
        X = np.array(df.drop(['label'], 1))
        y = np.array(df['label'])
        X = preprocessing.scale(X)
    i.e we dropped NaN and then preprocessed.
    Below we preprocess everything, also the last forecast_out (35) records,
    added X_lately with these records (which are now preprocessed, i.e. scaled) 
    and then dropped these records from X.
    .dropna() drops those 35 records from y
    """
    X = np.array(df.drop(['label'], axis=1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)
    y = np.array(df['label'])

    """
    # same as in three_regression()
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print("\n1.\n", confidence)

    """
    With pickle, you can save any Python object, like our classifier.
    After defining, training, and testing your classifier, add:
    """
    with open('linearregression.pickle', 'wb') as f:
        pickle.dump(clf, f)

    """
    # New part forecasting:
    # predict y on data X_lately; returns array with 35 numbers
    """
    forecast_set = clf.predict(X_lately)

    """
    # new column 'Forecast' filled with NaN
    # last_date: 
    #   df.iloc[-1].name: name property of last record, somehow generates a timestamp object
        print(last_date, type(last_date))
    #       Timestamp('2018-02-05 00:00:00') 
    #       pandas._libs.tslibs.timestamps.Timestamp
    # last_unix = 1517788800.0
    """
    df['Forecast'] = np.nan
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    """
    # take each forecast i of the 35 in the forecast_set
    # next_date:
    #   print(next_date, type(next_date))   
    #       datetime.datetime(2018, 2, 6, 1, 0)
    #       datetime.datetime
    # df.loc[next_date]
    #   fill all columns but the last with NaN and the last column with i
    #   analyze with:
    #       df.info()
    #       print(df.columns) output: Index(['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume', 'label', 'Forecast'], dtype='object')
    #       print(df.index.name) output: 'Date'
    #       print(len(df.columns)) output: 6
    #       print(range(len(df.columns) - 1)) output: range(0, 5)
    #       print([np.nan for _ in range(len(df.columns) - 1)] + [250.25]) output: [nan, nan, nan, nan, nan, 250.25]
    """
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    """
    # As you can see, we do not have labels for the last 35 days,
    # because each 'label' or a record, corresponds to the value 'Adj. Open' of a record 35 into the future
    # so there is no way to check, whether the prediction is good, because we don't have the data
    """
    df_display(df.tail(71), '2.')

    """
    # plot df columns 'Adj. Close' and 'Forecast'; index is column 'Date' (DatetimeIndex)
    """
    df['Adj. Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def five_pickle_loadTrainedClassifiers():
    """
    With pickle, you can save any Python object, like our classifier.
    After defining, training, and testing your classifier, add:
        with open('linearregression.pickle','wb') as f:
            pickle.dump(clf, f)
    Now, all you need to do to use the classifier is load in the pickle,
    save it to clf, and use just like normal. For example:
        pickle_in = open('linearregression.pickle','rb')
        clf = pickle.load(pickle_in)
    """
    from sklearn import preprocessing
    import pickle
    import datetime
    import quandl
    import math

    """
    # same as in four_regression_forecasts()
    """
    df = quandl.get("WIKI/GOOGL")
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

    forecast_col = 'Adj. Close'
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label'], axis=1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    df.dropna(inplace=True)

    """
    # commented out from four_regression_forecasts()
    """
    # y = np.array(df['label'])
    #
    # """
    # # same as in three_regression()
    # """
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # clf = LinearRegression(n_jobs=-1)
    # clf.fit(X_train, y_train)
    # confidence = clf.score(X_test, y_test)
    # print("\n1.\n", confidence)

    """
    # load trained classifier object
    """
    pickle_in = open('linearregression.pickle', 'rb')
    clf = pickle.load(pickle_in)

    """
    # same as in four_regression_forecasts()
    """
    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan

    """
    # same as in four_regression_forecasts()
    """
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    """
    # same as in four_regression_forecasts()
    """
    df['Adj. Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def six_linearRegression_programedFormulas():
    """
    linear regression:
    programed formulas
    """
    from statistics import mean

    xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

    def best_fit_slope_and_intercept(xs, ys):
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
             ((mean(xs) * mean(xs)) - mean(xs * xs)))
        b = mean(ys) - m * mean(xs)
        return m, b

    def squared_error(ys_orig, ys_line):
        return sum((ys_line - ys_orig) * (ys_line - ys_orig))

    def coefficient_of_determination(ys_orig, ys_line):
        y_mean_line = [mean(ys_orig) for y in ys_orig]
        squared_error_regr = squared_error(ys_orig, ys_line)
        squared_error_y_mean = squared_error(ys_orig, y_mean_line)
        return 1 - (squared_error_regr / squared_error_y_mean)

    m, b = best_fit_slope_and_intercept(xs, ys)
    regression_line = [(m * x) + b for x in xs]

    r_squared = coefficient_of_determination(ys, regression_line)
    print(r_squared)

    plt.scatter(xs,ys,color='#003F72',label='data')
    plt.plot(xs, regression_line, label='regression line')
    plt.legend(loc=4)
    plt.show()


def seven_creatingSampleData_forTesting():
    """
    Let's build a system that will generate example data, base on parameters we can choose:
        hm - The value will be "how much."
             This is how many datapoints that we want in the set.
             We could choose to have 10, or 10 million, for example.
        range - This will dictate how much each point can vary from the previous point.
                The more variance, the less-tight the data will be.
        slope - This will be how far to step on average per point, defaulting to 2.
        correlation - This will be either False, pos, or neg to indicate
                      that we want no correlation, positive correlation, or negative correlation.
    """
    from statistics import mean
    import random

    def create_dataset(hm, random_range, slope=2, correlation=False):
        val = 1
        ys = []
        half_random_range =  int(random_range / 2)
        for i in range(hm):
            y = val + random.randrange(-half_random_range, half_random_range)
            ys.append(y)
            if correlation == 'pos':
                val += slope
            elif correlation == 'neg':
                val -= slope
        xs = [i for i in range(len(ys))]
        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

    def best_fit_slope_and_intercept(xs, ys):
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
             ((mean(xs) * mean(xs)) - mean(xs * xs)))
        b = mean(ys) - m * mean(xs)
        return m, b

    def coefficient_of_determination(ys_orig, ys_line):
        y_mean_line = [mean(ys_orig) for y in ys_orig]
        squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
        squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))
        print(squared_error_regr)
        print(squared_error_y_mean)
        r_squared = 1 - (squared_error_regr / squared_error_y_mean)
        return r_squared

    """
    generate data and test
    """
    xs, ys = create_dataset(40, 40, 2, correlation='neg')
    m, b = best_fit_slope_and_intercept(xs, ys)
    regression_line = [(m * x) + b for x in xs]
    r_squared = coefficient_of_determination(ys, regression_line)
    print(r_squared)

    """
    plot
    """
    plt.scatter(xs, ys, color='#003F72', label='data')
    plt.plot(xs, regression_line, label='regression line')
    plt.legend(loc=4)
    plt.show()


def eight_applying_KNearestNeighbors_toData():
    """
    K-Nearest-Neighbors
    """
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn import neighbors
    import pandas as pd

    """
    # load data
    # handle missing values
    # drop column id
    """
    df = pd.read_csv('data/breast-cancer-wisconsin.data')
    df.drop(['id'], axis=1, inplace=True)

    missing_ = 2
    if missing_ == 0:
        df.replace('?', 0, inplace=True)
    elif missing_ == 1:
        df.replace('?', -99999, inplace=True)
    else:
        df.replace('?', np.nan, inplace=True)
        df.dropna(inplace=True)
    df.info()

    """
    # column 'class' is the label
    """
    X = np.array(df.drop(['class'], axis=1))
    y = np.array(df['class'])

    """
    # scaling, splitting and training
    """
    preprocessing_ = False
    if preprocessing_:
        X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    """
    # making up data and predicting
    # reshape takes the array and makes an array of this array, because clf.predict() wants that
    """
    example_X = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
    example_X = example_X.reshape(1, -1)
    if preprocessing_:
        example_X = preprocessing.scale(example_X)
    prediction = clf.predict(example_X)
    print(prediction)

    example_X = np.array([[4, 2, 1, 3, 1, 2, 3, 2, 1], [4, 2, 1, 1, 4, 2, 3, 2, 1], [4, 2, 4, 1, 1, 2, 3, 2, 1]])
    if preprocessing_:
        example_X = preprocessing.scale(example_X)
    prediction = clf.predict(example_X)
    print(prediction)


def nine_KNearestNeighbors_programedFormulas():
    """
    K-Nearest-Neighbors
    programed formulas
    """
    from collections import Counter
    import warnings
    import random
    import pandas as pd

    def k_nearest_neighbors(data, predict, k=3):
        if len(data) >= k:
            warnings.warn('K is set to a value less than total voting groups!')

        """
        # np.linalg.norm()
        # calculates the euclidian distance of predict to each group in the data
        # stores euclidian distance to each group 
        """
        distances = []
        for group in data:
            for features in data[group]:
                euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
                distances.append([euclidean_distance, group])

        """
        votes contains k=3 group labels: e.g. ['r', 'k', 'r']
        vote_results contains the 1 most common label: 
            most_common(1) returns: [('r', 3)]
            most_common(1)[0][0] returns: r (element 0 in array, element 0 in tuple)
        """
        votes = [i[1] for i in sorted(distances)[:k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1] / k
        return vote_result, confidence

    """
    # create labeled dataset
    # create unlabeled new data point
    """
    smalldata_ = False
    if smalldata_:
        dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
        new_datapoint = [5, 3]

        """
        # scatterplot the dataset:
        # same as:
        # for i in dataset:
        #     for ii in dataset[i]:
        #         plt.scatter(ii[0],ii[1],s=100,color=i)
        """
        [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]

        """
        # run algorithm and plot result
        # result is k or r
        # scatterplot the result
        """
        result, confidence = k_nearest_neighbors(dataset, new_datapoint)
        plt.scatter(new_datapoint[0], new_datapoint[1], s=300, color=result)
        plt.show()

    else:
        """
        # load the dataset, handle missing values, drop column id
        """
        df = pd.read_csv('data/breast-cancer-wisconsin.data')
        df.replace('?', -99999, inplace=True)
        df.drop(['id'], axis=1, inplace=True)

        """
        # what train_test_split() is doing:
        #   shuffle the dataset
        #   define train_set and test_set dicts: 2 and 4 are the labels
        #   split dataset into train_data and test_data
        """
        full_data = df.astype(float).values.tolist()
        random.shuffle(full_data)
        test_size = 0.2
        train_set = {2: [], 4: []}
        test_set = {2: [], 4: []}
        train_data = full_data[:-int(test_size * len(full_data))]
        test_data = full_data[-int(test_size * len(full_data)):]

        """
        # fill train_data into train_set: record by record; i[-1] is the label of the dict
        # fill test_data into test_set: record by record; i[-1] is the label of the dict
        # train_set = {2: [], 4: []}
        # test_set = {2: [], 4: []}: 
        # so e.g. test_set[2].append([3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0]) 
        # will put the list i[:-1] at the end of label 2 in the dict
        """
        for i in train_data:
            train_set[i[-1]].append(i[:-1])
        for i in test_data:
            test_set[i[-1]].append(i[:-1])

        """
        label: in dict test_set is first assigned 2 and then 4
        data:  in dict test_set is assigned one record after the other in group 2 
               and then one record after the other in group 4
        k_nearest_neighbors: gets one record after the other together with the train_set
                             returns a vote which is either 2 or 4
        if the vote equals the label correct is incremented
        """
        correct = 0
        total = 0
        confidence_ = 0
        for label in test_set:
            for data in test_set[label]:
                vote, confidence = k_nearest_neighbors(train_set, data, k=5)
                if label == vote:
                    correct += 1
                else:
                    print("1. ", confidence, vote, label, data)
                total += 1
                if confidence < 1:
                    print("2. ", confidence,  vote, label, data)
                confidence_ = confidence_ + confidence
        print('Accuracy:', correct / total)
        print('Confidence:', confidence_ / total)


def ten_supportVectorMachine():
    """
    Support Vector Machine
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    import pandas as pd

    """
    # load the dataset, handle missing values, drop column id
    """
    df = pd.read_csv('data/breast-cancer-wisconsin.data')
    if 1:
        df.replace('?', -99999, inplace=True)
    else:
        df.replace('?', 0, inplace=True)

    df.drop(['id'], axis=1, inplace=True)

    """
    # column 'class' is the label
    """
    X = np.array(df.drop(['class'], axis=1))
    y = np.array(df['class'])

    """
    # scaling, splitting and training
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if 0:
        clf = svm.SVC()
    else:
        clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    """
    # making up data and predicting
    # reshape takes the array and makes an array of this array, because clf.predict() wants that
    """
    example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])
    prediction = clf.predict(example_measures)
    print(prediction)


def eleven_SupportVectorMachine_programedFormulas():
    """
    Support Vektor Machine
    programed formulas
    The convex optimization problem could be solved with packages like:
        cvxopt or libsvm, which sklearn / scikit-learn uses
    """
    class Support_Vector_Machine:
        """
        Our SVM class
        SVM is a binary classifier: it separates only into two groups at a time.
                                    it separates one at a time from the rest.
                                    the two groups are denoted as positive and negative or 1 and -1.
                                    aim: - find the decision boundary, i.e. the best separating hyperplane.
                                         - in two dimensions, i.e. with two features, the hyperplane is a straight line.
                                         - the separating hyperplane will be there,
                                           where the distances to the features are maximal.
                                    prediction: with the hyperplane new unknown data can then easily be separated
                                                into one of the two groups.
        """
        """
        # The __init__ method of a class is one that runs whenever an object is created with the class. 
        # The other methods will only run when called to run. 
        # For every method, we pass "self" as the first parameter mainly out of standards. 
        # Next, we are adding a visualization parameter. 
        # We're going to want to see the SVM most likely, so we're setting that default to true. 
        """
        def __init__(self, visualization=True):
            """
            called when Support_Vector_Machine object is created
            """
            """
            # You can see some variables like self.color and self.visualization. 
            # Doing this will allow us to reference self.colors for example in other methods within our class. 
            """
            self.visualization = visualization
            self.colors = {1: 'r', -1: 'b'}
            """
            # if we have visualization turned on, we're going to begin setting up our graph.
            """
            if self.visualization:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(1, 1, 1)

        """
        # train
        """
        def fit(self, data):
            self.data = data

            """
            # { ||w||: [w,b] }
            """
            opt_dict = {}

            """
            # our intention is to make sure, we check every version of the vector possible.
            """
            transforms = [[1, 1],
                          [-1, 1],
                          [-1, -1],
                          [1, -1]]

            """
            # we need some starting point that matches our data. 
            # To do this, we're going to first reference our training data 
            # to pick some haflway decent starting values:
            # take the data
            #    take one record
            #        take one feature
            #            append it to all_data
            # get max and min feature values
            """
            all_data = []
            for yi in self.data:
                for feature_set in self.data[yi]:
                    for feature in feature_set:
                        all_data.append(feature)
            self.max_feature_value = max(all_data)
            self.min_feature_value = min(all_data)
            all_data = None

            """
            # define step sizes in the feature space
            # For our first pass, we'll take big steps (10%). 
            # Once we find the minimum with these steps, 
            # we're going to step down to a 1% step size to continue finding the minimum here. 
            # Then, one more time, we step down to 0.1% for fine tuning. 
            # We could continue stepping down, depending on how precise you want to get.
            """
            step_sizes = [self.max_feature_value * 0.1,
                          self.max_feature_value * 0.01,
                          # point of expense:
                          self.max_feature_value * 0.001, ]

            """
            # we're going to set some variables that will help us make steps with b 
            # (used to make larger steps than we use for w, since we care far more about w precision than b), 
            # and keep track of the latest optimal value
            #   extremely expensive
            #   we dont need to take as small of steps
            #   with b as we do w
            """
            b_range_multiple = 5
            b_multiple = 5
            latest_optimum = self.max_feature_value * 10

            """
            # Now we're ready to begin stepping:
            # The idea here is to begin stepping down the vector. 
            # To begin, we'll set optimized to False, and we'll reset this for each major step. 
            # The optimized var will be true 
            # when we have checked all steps down to the base of the convex shape (our bowl).
            """
            for step in step_sizes:

                """
                # start value for w for each step
                """
                w = np.array([latest_optimum, latest_optimum])

                """
                # we can do this because optimization problem is convex
                """
                optimized = False
                while not optimized:
                    """
                    # Here, we begin also iterating through possible b values, 
                    and now you can see our b values we set earlier in action. 
                    I will note here that we're straight iterating through b with a constant step-size. 
                    We could also break down b steps just like we did with w. 
                    To make things more accurate and precise, you probably would want to implement that. 
                    That said, I am going to skip doing that for brevity, 
                    since we'll achieve similar results either way and we're not trying to win any awards here.
                    """
                    """
                    # arange(start, stop, step) 
                    #   Values are generated within the half-open interval [start, stop), 
                    #   with spacing between values given by step.
                    """
                    for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                       self.max_feature_value * b_range_multiple,
                                       step * b_multiple):

                        """
                        # try out all four transformations for w
                        """
                        for transformation in transforms:
                            w_t = w * transformation
                            found_option = True

                            """
                            # weakest link in the SVM fundamentally
                            # SMO attempts to fix this a bit
                            # yi(xi.w+b) >= 1
                            #
                            # I commented in a suggestion for a break here. 
                            # If just one variable doesn't work you might as well give up on the rest
                            # since just 1 that doesn't fit is enough to toss the values for w and b. 
                            # You could break there, as well as in the preceeding for loop. For now, 
                            # I will leave the code as I originally had it, 
                            # but I thought of the change whenever I was filming the video version.
                            """
                            """
                            # our goal is to: 
                            #   minimize ||w||, 
                            #   maximize b, 
                            #   with the constraint such that yi(xi.w+b) >= 1
                            # from data
                            #   key: i (yi)
                            #   value: xi
                            # check whether w_t and b do or do NOT satisfy constraint equation
                            """
                            for i in self.data:
                                for xi in self.data[i]:
                                    yi = i
                                    if not yi * (np.dot(w_t, xi) + b) >= 1:
                                        found_option = False

                            """
                            # store each found option in opt_dict as key=norm(w_t) value=[w_t, b] 
                            """
                            if found_option:
                                opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                    """
                    # here we decide whether or not the step is finished
                    # condition for step optimized: w[0] becomes negative
                    # why? because: 
                    # Once we've passed zero with our stepping of the w vector, 
                    # there's no reason to continue since we've tested the negatives via the transformation, 
                    # thus we'll be done with that step size and either continue to the next, 
                    # or be done entirely. If we've not passed 0, then we take another step.
                    """
                    if w[0] < 0:
                        optimized = True
                        print('Optimized a step.')
                    else:
                        w = w - step

                """
                # equation:||w|| : [w,b]
                # sort the key = norm(w)
                # we are interested in the vector [w,b] with the smallest norm, i.e norms[0]
                """
                norms = sorted([n for n in opt_dict])
                opt_choice = opt_dict[norms[0]]
                self.w = opt_choice[0]
                self.b = opt_choice[1]

                """
                # we set latest optimums, 
                #   and we either may take another step 
                #   or be totally done with the entire process (if we have no more steps to take).
                """
                latest_optimum = opt_choice[0][0] + step * 2

        """
        # predict label
        """
        def predict(self, features):
            """
            # classification is just:
            # sign(xi.w+b)
            """
            classification = np.sign(np.dot(np.array(features), self.w) + self.b)

            """
            # if the classification isn't zero, and we have visualization on, we graph
            # We're just going to do one at a time, 
            # but you could augment the code to do many at once like scikit-learn does.
            """
            if classification != 0 and self.visualization:
                self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
            else:
                print('feature_set', features, 'is on the decision boundary')
            return classification

        """
        # visualize everything
        """
        def visualize(self):
            """
            visualize feature sets, hyperplanes
            """
            """
            # scattering known feature_sets.
            # All that one liner is doing is going through our data 
            # and graphing it along with its associated color. 
            # See the video if you want to see it more broken down.
            """
            [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

            """
            # hyperplane = x.w+b
            # v = x.w+b
            # psv = 1
            # nsv = -1
            # dec = 0
            """
            """
            # We want to graph our hyperplanes for the positive and negative support vectors, 
            # along with the decision boundary. 
            # In order to do this, we need at least two points for each to create a "line" 
            # which will be our hyperplane.
            # Once we know what w and b are, 
            # we can use algebra to create a function 
            # that will return to us the value needed for our second feature (x2) to make the line:
            """
            def hyperplane(x, w, b, v):
                return (-w[0] * x - b + v) / w[1]

            """
            we create some variables to house various data that we're going to reference
            """
            data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
            hyp_x_min = data_range[0]
            hyp_x_max = data_range[1]

            """
            # (w.x+b) = 1
            # positive support vector hyperplane
            """
            psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
            psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
            self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

            """
            # (w.x+b) = -1
            # negative support vector hyperplane
            """
            nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
            nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
            self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

            """
            # (w.x+b) = 0
            # positive support vector hyperplane
            """
            db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
            db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
            self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

            plt.show()

    """
    # made up data
    # key: yi (label)
    # value: xi (features)
    """
    data_dict = {-1: np.array([[1, 7],
                               [2, 8],
                               [3, 8], ]),

                 1: np.array([[5, 1],
                              [6, -1],
                              [7, 3], ])}

    """
    # fit data with own class Support_Vector_Machine
    """
    svm = Support_Vector_Machine()
    svm.fit(data=data_dict)

    """
    # data x, for which we want to predict label y
    """
    predict_us = [[0, 10],
                  [1, 3],
                  [3, 4],
                  [3, 5],
                  [5, 5],
                  [5, 6],
                  [6, -5],
                  [5, 8]]

    """
    # call method predict() from our class Support_Vector_Machine
    """
    for p in predict_us:
        svm.predict(p)

    """
    # call method visualize() from our class Support_Vector_Machine
    """
    svm.visualize()


def twelve_kernels_softMarginSVM_CVXOPT(testalgorithm):
    """
    Kernels, Soft Margin SVM, and Quadratic Programming with Python and CVXOPT
    # Mathieu Blondel, September 2010
    # License: BSD 3 clause
    # http://www.mblondel.org/journal/2010/09/19/support-vector-machines-in-python/
    # visualizing what translating to another dimension does
    # and bringing back to 2D:
    # https://www.youtube.com/watch?v=3liCbRZPrZA
    # Docs: http://cvxopt.org/userguide/coneprog.html#quadratic-programming
    # Docs qp example: http://cvxopt.org/examples/tutorial/qp.html
    # Nice tutorial:
    # https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    """
    import numpy as np
    from numpy import linalg
    import cvxopt
    import cvxopt.solvers

    """
    linear_kernel: just the original svm, where the dot-product is now also identified as kernel
    polynomial_kernel: hyperplanes can get waved
    gaussian kernel: hyperplanes can get curved
    """
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(x, y, p=3):
        return (1 + np.dot(x, y)) ** p

    def gaussian_kernel(x, y, sigma=5.0):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

    """
    # SVM inherits from object
    # used in __init__() to feed 
    #   kernel into self.kernel = kernel
    #   C into self.C = C
    """
    class SVM(object):
        """
        # support vector machine implemented for the use of different kernels
        """
        """
        # initialisation called when object is built
        # to set: self.kernel and self.C
        """
        def __init__(self, kernel=linear_kernel, C=None):
            self.kernel = kernel
            self.C = C
            if self.C is not None: self.C = float(self.C)

        """
        # svm training algorithm
        """
        def fit(self, X, y):
            """
            programed formulas
            X: features
            y: labels / groups
            """
            """
            # rows, columns of feature vector X
            """
            n_samples, n_features = X.shape

            """
            # Gram matrix
            # filled by repeatedly calling calculated kernel function:
            #   for each sample in X, i.e X[i], 
            #       calculate the kernel function with each other sample in X, i.e. X[j]
            # K:
            # [[ 1.26126396  2.08589044  1.698127   ... -1.77613142 -1.64989031  -1.47657789]
            #  [ 2.08589044 12.20800789  8.16087848 ...  4.11673069  2.57224848 -3.29564812]
            #  [ 1.698127    8.16087848  5.55738772 ...  1.9196632   1.01815717 -2.50972343] 
            #  ...
            #  [-1.77613142  4.11673069  1.9196632  ...  8.18267582  6.59279572  1.39178031]
            #  [-1.64989031  2.57224848  1.01815717 ...  6.59279572  5.3665222  1.41487733]
            #  [-1.47657789 -3.29564812 -2.50972343 ...  1.39178031  1.41487733 1.81185522]]  
            """
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = self.kernel(X[i], X[j])

            """
            # needed matrices for the convex optimization with cvxopt
            # K is the kernel matrix, y the label vector, n_samples the sample size
            """
            P = cvxopt.matrix(np.outer(y, y) * K)
            q = cvxopt.matrix(np.ones(n_samples) * -1)
            A = cvxopt.matrix(y, (1, n_samples))
            b = cvxopt.matrix(0.0)

            """
            # two more matrices needed matrices, whose values depend on input C
            """
            if self.C is None:
                G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
                h = cvxopt.matrix(np.zeros(n_samples))
            else:
                tmp1 = np.diag(np.ones(n_samples) * -1)
                tmp2 = np.identity(n_samples)
                G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
                tmp1 = np.zeros(n_samples)
                tmp2 = np.ones(n_samples) * self.C
                h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

            """
            # solve QP problem
            # solution returns: 
            #  {'x': <180x1 matrix, tc='d'>, 'y': <1x1 matrix, tc='d'>, 
            #  's': <180x1 matrix, tc='d'>, 'z': <180x1 matrix, tc='d'>, 
            #  'status': 'optimal', 
            #  'gap': 3.384698000599809e-08, 'relative gap': 2.237957430996584e-08, 
            #  'primal objective': -1.5124049965028024, 'dual objective': -1.5124050291026703, 
            #  'primal infeasibility': 6.084076112416481e-11, 'dual infeasibility': 4.8392702128758635e-12, 
            #  'primal slack': 6.2588139870831e-11, 'dual slack': 5.793620029378609e-10, 
            #  'iterations': 7}
            # 
            #      pcost       dcost       gap    pres   dres
            #  0: -1.6786e+01 -2.9904e+01  5e+02  2e+01  2e+00
            #  1: -1.9288e+01 -9.1514e+00  1e+02  5e+00  4e-01
            #  2: -1.0654e+01 -4.5186e+00  6e+01  2e+00  2e-01
            #  3: -2.4170e+00 -2.8136e+00  1e+00  3e-02  2e-03
            #  4: -2.5554e+00 -2.6327e+00  2e-01  5e-03  4e-04
            #  5: -2.5847e+00 -2.6030e+00  5e-02  9e-04  7e-05
            #  6: -2.5983e+00 -2.5987e+00  7e-04  9e-06  7e-07
            #  7: -2.5986e+00 -2.5986e+00  7e-06  9e-08  7e-09
            #  8: -2.5986e+00 -2.5986e+00  7e-08  9e-10  7e-11
            """
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)

            """
            # Lagrange multipliers
            # stored into a numpy array
            # a:
            # [7.52068394e-11 6.62357981e-11 7.56744428e-11 1.02277675e-10
            #  8.18797220e-11 7.48985822e-11 1.37956747e-10 7.11059948e-11
            #  1.00198375e-10 6.46328852e-11 6.62161436e-11 7.29768264e-11
            #  8.18740729e-11 8.87636716e-11 1.44599354e-10 6.90444018e-11
            #  7.01155519e-11 8.09842087e-11 9.25210158e-11 7.19534356e-11
            #  7.76497745e-11 6.27758931e-10 2.44501888e-10 7.42236979e-11
            #  6.10940907e-11 1.80284219e-10 4.16155943e-10 8.93854073e-11
            #  6.24651607e-11 7.21952482e-11 7.68991579e-11 1.66614691e-10
            #  1.06614912e-10 7.41395937e-11 9.15150720e-11 1.02651280e-10
            #  8.02273229e-11 7.00637482e-11 6.74434482e-11 6.81754067e-11
            #  8.10892016e-11 6.56022116e-11 2.69841163e-10 6.99630057e-11
            #  6.80572751e-11 1.56852876e-10 6.55763703e-11 7.33097720e-11
            #  1.81521306e-10 1.42413885e-10 8.57331521e-11 8.83058556e-11
            #  1.64893115e-10 4.23020800e-01 6.67772902e-11 7.50998839e-11
            #  1.03794043e-10 1.49669318e-10 1.08938420e+00 7.56022619e-11
            #  6.43925103e-11 1.05201003e-10 7.08321852e-11 6.80253012e-11
            #  1.45500214e-10 9.75730691e-11 8.09855923e-11 1.55418513e-10
            #  1.98960491e-10 7.51363163e-11 6.77669922e-11 6.81370461e-11
            #  6.10413399e-11 9.92155237e-11 4.93325926e-10 1.11224867e-10
            #  1.13683226e-10 1.36476987e-10 7.78919824e-09 7.06503402e-11
            #  6.89600141e-11 8.34468428e-11 6.56187474e-11 7.01075347e-11
            #  6.99292174e-11 6.96321411e-11 1.32117314e-10 7.47108219e-11
            #  8.87860625e-11 7.22073860e-11 5.07876504e-10 7.89269665e-11
            #  9.99443532e-11 6.91188122e-11 1.33033451e-09 1.09900572e-10
            #  2.66436273e-10 8.63134939e-11 7.06245728e-11 7.57010398e-11
            #  1.22328181e-10 2.93265777e-10 9.03975537e-11 3.00093803e-10
            #  7.70515806e-11 7.58941132e-11 7.25936939e-11 7.62716591e-11
            #  2.32355115e-09 1.35135804e-10 2.37576101e-10 2.85220132e-09
            #  7.72126860e-11 4.33900592e-10 6.77369674e-11 9.31309128e-11
            #  7.30223936e-11 6.76128630e-11 1.62042136e-09 1.28220441e-10
            #  7.94709178e-11 6.88609885e-11 8.02804643e-11 5.80533369e-11
            #  7.89394618e-11 1.51240499e+00 4.19474591e-10 9.14142298e-11
            #  3.31945593e-10 7.92821312e-11 9.86472705e-11 8.28422411e-11
            #  7.40860017e-11 7.70448426e-11 8.49734070e-11 1.65584736e-10
            #  1.83644398e-10 2.11160759e-09 7.70427527e-11 8.21608889e-11
            #  1.04618534e-10 7.37036458e-11 6.18973719e-11 8.55842459e-11
            #  2.56906647e-10 6.93507090e-11 7.95161431e-11 8.01596708e-11
            #  2.95768528e-10 8.17472173e-11 6.83207292e-11 9.21278589e-11
            #  2.65217672e-10 7.55291463e-11 8.09724898e-11 7.65342816e-11
            #  6.83949605e-11 6.55799740e-11 1.38986086e-09 1.82610880e-10
            #  4.00758228e-09 8.18223707e-11 6.10639698e-11 7.83806977e-11
            #  6.31646959e-11 7.34357855e-11 1.95315279e-10 1.02719343e-10
            #  7.55352017e-11 9.50865223e-11 9.93492545e-11 9.23691773e-11
            #  8.17136804e-11 8.72714150e-11 6.81368983e-11 7.86511401e-11
            #  1.77755016e-10 1.51505716e-10 8.62477721e-11 1.02593960e-10]
            """
            a = np.ravel(solution['x'])

            """
            # Support vectors:
            #   self.sv
            # 1. sv = a > 1e-5
            #    for each element in a that is > 1e-5 set True, else set False   
            #  sv:  
            #  [False False False False False False False False False False False False
            #  False False False False False False False False False False False False
            #  False False False False False False False False False False False False
            #  False False False False False False False False False False False False
            #  False False False False False  True False False False False  True False
            #  False False False False False False False False False False False False
            #  False False False False False False False False False False False False
            #  False False False False False False False False False False False False
            #  False False False False False False False False False False False False
            #  False False False False False False False False False False False False
            #  False False False False False  True False False False False False False
            #  False False False False False False False False False False False False
            #  False False False False False False False False False False False False
            #  False False False False False False False False False False False False
            #  False False False False False False False False False False False False]
            # 2. ind = np.arange(len(a))[sv]
            #    for array [0, 1, ... 179] pick those, at the position, where the filter vector [sv] is True
            #       ind:  [ 53  58 125]
            # 3. self.a : [0.4230208  1.0893842  1.51240499]
            # 4. self.sv :  [[0.21466427 1.01407065]
            #                [1.65925527 2.61380914]
            #                [2.10867558 1.39566005]]
            # 6. self.sv_y: [ 1.  1. -1.]
            """
            sv = a > 1e-5
            ind = np.arange(len(a))[sv]
            self.a = a[sv]
            self.sv = X[sv]
            self.sv_y = y[sv]
            print("%d support vectors out of %d points" % (len(self.a), n_samples))

            """
            # Intercept
            # calculation of b:
            #   q += r -> q = q + r; q -= r -> q = q - r; q /= r -> q = q / r
            #   self.b = (sum_len_a(yi) - sum_len_a(sum(a*y*K[ind_i, X]))) / len(a)
            # self.sv_y : [ 1.  1. -1.]
            # self.a: [0.4230208  1.0893842  1.51240499]
            # self.a * self.sv_y : [0.4230208  1.0893842  -1.51240499]
            # ind : [ 53  58 125]
            # K[ind[n], sv] : [7.38039483 0.48174164 6.51267837]
            #   row: ind[n]; columns: where in sv the elements are True, False is filtered out
            # np.sum(self.a * self.sv_y * K[ind[n], sv]) : scalar
            """
            self.b = 0
            for n in range(len(self.a)):
                self.b += self.sv_y[n]
                self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
            self.b /= len(self.a)

            """
            # Weight vector
            # calculation of w for the linear kernel; None for non-linear kernel
            # self.a : [0.4230208  1.0893842  1.51240499]
            # self.sv_y: [ 1.  1. -1.]
            # self.sv :  [[0.21466427 1.01407065]
            #             [1.65925527 2.61380914]
            #             [2.10867558 1.39566005]]
            # self.a[n] * self.sv_y[n] * self.sv[n] :
            #       0.4230208 * 1. * [0.21466427 1.01407065] = [0.09080745, 0.42897298]
            #       1.0893842 * 1. * [1.65925527 2.61380914] = [1.80756647, 2.84744238]
            #       1.51240499*-1. * [2.10867558 1.39566005] = [-3.18917147, -2.11080322]
            # self.w : [-1.29079755,  1.16561214]
            """
            if self.kernel == linear_kernel:
                self.w = np.zeros(n_features)
                for n in range(len(self.a)):
                    self.w += self.a[n] * self.sv_y[n] * self.sv[n]
            else:
                self.w = None

        """
        # support vector hyperplanes
        # linear case:
        #    X*w+b=-1, X*w+b=0, X*w+b=1
        # else:
        #    y_predict + b  
        #
        # X:  
        # [[-0.30223955  1.18181808]
        #  [ 0.79023786  1.76756176]
        #  [ 0.52262182  2.84984729]
        #  [-0.30421905  0.82657608]
        #  [-1.35580552  1.32457406]
        #  [-0.32952018  0.83591436]
        #  [-0.49660891  1.27614129]
        #  [ 0.42277639  1.63285323]
        #  [ 0.56704374  3.41282202]
        #  [ 0.84638558  2.44024155]
        #  [ 1.82937863 -1.31218553]
        #  [ 2.3211248   0.39477735]
        #  [ 1.32773988 -0.44374365]
        #  [ 0.61312012 -1.61424381]
        #  [ 0.96815189 -0.69259025]
        #  [ 1.18809566 -0.41617538]
        #  [ 1.50580216 -0.89077922]
        #  [ 1.0869791  -0.33425179]
        #  [ 1.45566943 -0.10297959]
        #  [ 1.08308638  0.0998061 ]]
        # w:  
        #  [-1.56828713  1.79622257]
        # b:  0.20844348234185736
        # X.w+b:  
        # [ 2.80525019  2.14405796  4.50778244  2.17026093  4.71396565  2.22671399
        #   3.27950263  2.47837654  5.44934405  3.26428482 -5.01752476 -2.72263868
        #  -2.67089623 -3.65264609 -2.55394291 -2.40237526 -3.7531244  -2.09664245
        #  -2.25943841 -1.31087296] 
        # first sample:
        #   In[15]:  np.dot(np.array([-0.30223955,  1.18181808]),np.array([-1.56828713,  1.79622257]))+0.20844348234185736
        #   Out[15]: 2.8052501877139147
        #   positive sign -> group +1
        #
        # self.a :    [0.4230208  1.0893842  1.51240499]
        # self.sv_y : [ 1.  1. -1.]
        # self.sv :  [[0.21466427 1.01407065]
        #             [1.65925527 2.61380914]
        #             [2.10867558 1.39566005]]
        """
        def project(self, X):
            if self.w is not None:
                return np.dot(X, self.w) + self.b
            else:
                y_predict = np.zeros(len(X))
                """
                # iterate with i over all 20 test samples
                # y_predict[i]:
                #   0.4230208 * 1.0 * kernel(X[i],[0.21466427 1.01407065]) +
                #   1.0893842 * 1.0 * kernel(X[i],[1.65925527 2.61380914]) +
                #   1.51240499 * -1.0 * kernel(X[i],[2.10867558 1.39566005])
                # function also called from plot_contour(): 
                #   i iterates from 0 to 2499 for all point of a mesh within -6,6;-6,6
                #   and it calculates y_predict+self.b for each point in the mesh
                #   it then finds the hyperplanes where the value is -1, 0, 1 by interpolation in contour()
                """
                for i in range(len(X)):
                    s = 0
                    for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                        s += a * sv_y * self.kernel(X[i], sv)
                    y_predict[i] = s
                return y_predict + self.b

        """
        # the prediction of the label only depends on the sign - / + of the project method
        """
        def predict(self, X):
            return np.sign(self.project(X))


    #if __name__ == "__main__":
    import pylab as pl

    """
    # generate data that is linearly separable
    """
    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    """
    # generate data that is not linearly separable
    """
    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0, 0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    """
    # generate data that is linearly separable, but overlaps
    """
    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1

        """
        dump and load array
        """
        import pickle
        # version 1 with pickle
        filename = 'np_arrays_pickle_dump__gen_lin_separable_overlap_data.pkl'
        if 0:
            with open(filename, 'wb') as f: pickle.dump(obj={'X1': X1, 'y1': y1, 'X2': X2, 'y2': y2,}, file=f)
        elif 0:
            with open(filename, 'rb') as f: dict_ = pickle.load(file=f)
            X1 = dict_["X1"]
            y1 = dict_["y1"]
            X2 = dict_["X2"]
            y2 = dict_["y2"]
            print("load: ", X1.shape, y1.shape, X2.shape, y2.shape)

        # version 2 with np.savez
        filename = 'np_arrays_np_savez__gen_lin_separable_overlap_data.npz'
        if 0:
            np.savez(filename, X1=X1, y1=y1, X2=X2, y2=y2)
        elif 1:
            pass
            dict_ = np.load(filename)
            X1 = dict_["X1"]
            y1 = dict_["y1"]
            X2 = dict_["X2"]
            y2 = dict_["y2"]

        """
        return
        """
        return X1, y1, X2, y2

    """
    # split 90% of data into training set
    """
    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    """
    # split 10% of data into test set
    """
    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    """
    # plot training samples and hyperplanes / decision boundary 
    # called from test_linear() only
    """
    def plot_margin(X1_train, X2_train, X1_test_c, X1_test_f, X2_test_c, X2_test_f, clf):
        """
        1. plot 90 blue and red training samples each
        2. plot 3 support vectors in green (blue, red still visible)
        3. plot hyperplane at 0 black solid
        4. plot hyperplanes at -1, 1 in black dashed
        """
        """
        # given x0, return x1 such that [x0,x1] is on the line
        # w.x + b = c
        # [w0, w1].[x0,x1]+b=c; w0*x0+w1*x1+b=c; w1*x1=c-b-w0*x0; x1=-w0*x0-b+c
        """
        def f(x0, w, b, c=0):
            x1 = (-w[0] * x0 - b + c) / w[1]
            return x1

        """
        # 1. ro: plot 90 training samples as red circle
        #    bo: plot 90 training samples as blue circle
        # 2. g:  scatter  3 support vectors as green with size 100
        """
        pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

        """
        # plot test predictions
        """
        pl.plot(X1_test_c[:, 0], X1_test_c[:, 1], "rx")
        pl.plot(X1_test_f[:, 0], X1_test_f[:, 1], "yx")
        pl.plot(X2_test_c[:, 0], X2_test_c[:, 1], "bx")
        pl.plot(X2_test_f[:, 0], X2_test_f[:, 1], "kx")


        """
        # 3. w.x + b = 0
        #   k: black line
        #   [a0, b0] : from to on horizontal axis 
        #   [a1, b1] : from to on vertical axis 
        """
        a0 = -4
        a1 = f(a0, clf.w, clf.b)
        b0 = 4
        b1 = f(b0, clf.w, clf.b)
        pl.plot([a0, b0], [a1, b1], "k")

        """
        # 4. w.x + b = 1
        #    k--: black line
        #   [a0, b0] : from to on horizontal axis 
        #   [a1, b1] : from to on vertical axis 
        """
        a0 = -4
        a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4
        b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0, b0], [a1, b1], "k--")

        """
        # 4. w.x + b = -1
        #    k--: black line
        #   [a0, b0] : from to on horizontal axis 
        #   [a1, b1] : from to on vertical axis 
        """
        a0 = -4
        a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4
        b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0, b0], [a1, b1], "k--")

        pl.axis("tight")
        pl.show()

    """ 
    # plot training samples and hyperplanes / decision boundary
    # called from test_non_linear(), test_non_linear_2(), test_soft()
    """
    def plot_contour(X1_train, X2_train, X1_test_c, X1_test_f, X2_test_c, X2_test_f, clf):
        """
        1. plot 90 blue and red training samples each
        2. plot 3 support vectors in green (blue, red still visible)
        3. plot hyperplane at 0 black solid
        4. plot hyperplanes at -1, 1 in black dashed
        """
        """
        # 1. ro: plot 90 training samples as red circle
        #    bo: plot 90 training samples as blue circle
        # 2. g:  scatter 8-10 support vectors as green with size 100
        """
        pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

        """
        # plot test predictions
        """
        pl.plot(X1_test_c[:, 0], X1_test_c[:, 1], "rx")
        pl.plot(X1_test_f[:, 0], X1_test_f[:, 1], "yx")
        pl.plot(X2_test_c[:, 0], X2_test_c[:, 1], "bx")
        pl.plot(X2_test_f[:, 0], X2_test_f[:, 1], "kx")

        """
        np.linspace(): returns 50 evenly spaced points from -6 to 6
        np.meshgrid(): 50 x 50 matrix in x1 between -6,6 and x2 between -6,6
        X1: is horizontally repeating np.linspace(); shape (50,50)
        X2: is vertically repeating np.linspace(); shape (50,50)
        X:  vector with shape (2500,2) containing all 2500 meshgrid points (x0, x1) 
        Z:  what clf.project(X).reshape(X1.shape) returns; shape (50,50) because of the reshape to X1.shape (50,50)
        """
        X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)

        """
        # pylab.contour()
        # k: black line; decision boundary Z; pl.contour(X1, X2, Z, [0.0]): plot points (x1,x2) where Z is [0.0]
        # grey: grey line; hyperplane Z+1; pl.contour(X1, X2, Z+1, [0.0]): plot points (x1,x2) where Z+1 is [0.0]
        # grey: grey line; hyperplane Z-1; pl.contour(X1, X2, Z-1, [0.0]): plot points (x1,x2) where Z-1 is [0.0]
        # - you won't find the exact values [0.0] in the Z, Z+1 or Z-1 matrix. 
        #   Contour is going to interpolate, where signs between subsequent values change from - to +
        """
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    """
    # linear test function
    """
    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        #plot_margin(X_train[y_train == 1], X_train[y_train == -1], clf)
        plot_margin(X_train[y_train == 1], X_train[y_train == -1],
                    X_test[(y_test == 1) & (y_predict == 1)], X_test[(y_test == 1) & (y_predict == -1)],
                    X_test[(y_test == -1) & (y_predict == -1)], X_test[(y_test == -1) & (y_predict == 1)], clf)

    """
    # non linear test function
    """
    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(polynomial_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        #plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf)
        plot_contour(X_train[y_train == 1], X_train[y_train == -1],
                    X_test[(y_test == 1) & (y_predict == 1)], X_test[(y_test == 1) & (y_predict == -1)],
                    X_test[(y_test == -1) & (y_predict == -1)], X_test[(y_test == -1) & (y_predict == 1)], clf)

    def test_non_linear_2():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(gaussian_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        #plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf)
        plot_contour(X_train[y_train == 1], X_train[y_train == -1],
                     X_test[(y_test == 1) & (y_predict == 1)], X_test[(y_test == 1) & (y_predict == -1)],
                     X_test[(y_test == -1) & (y_predict == -1)], X_test[(y_test == -1) & (y_predict == 1)], clf)

    """
    # soft test function for linearly separable data, that overlaps
    """
    def test_soft():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(C=15)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        #plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf)
        plot_contour(X_train[y_train == 1], X_train[y_train == -1],
                     X_test[(y_test == 1) & (y_predict == 1)], X_test[(y_test == 1) & (y_predict == -1)],
                     X_test[(y_test == -1) & (y_predict == -1)], X_test[(y_test == -1) & (y_predict == 1)], clf)

    """
    # soft test function for linearly separable data, that overlaps
    """
    if testalgorithm == "test_linear":
        test_linear()
    elif testalgorithm == "test_non_linear":
        test_non_linear()
    elif testalgorithm == "test_non_linear_2":
        test_non_linear_2()
    elif testalgorithm == "test_soft":
        test_soft()
    else:
        print("choose either: test_linear, test_non_linear, test_non_linear_2 or test_soft")


def thirteen_Clustering_KMeans():
    """
    KMeans clustering
    simple example
    """
    from sklearn.cluster import KMeans

    """
    # sample data
    """
    X = np.array([[1, 2],
                  [1.5, 1.8],
                  [5, 8],
                  [8, 8],
                  [1, 0.6],
                  [9, 11]])
    plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=5, zorder=10)
    plt.show()

    """
    # KMeans, K=2
    """
    clf = KMeans(n_clusters=2)
    clf.fit(X)

    """
    # get found centers and labels
    """
    centroids = clf.cluster_centers_
    labels = clf.labels_

    """
    # plot samples and KMeans found centers 
    """
    colors = ["g.", "r.", "c.", "y."]
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    plt.show()


def fourteen_Clustering_KMeans_TitanicData():
    """
    KMeans
    Real world example: Titanic (https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls)
    Dataset:
        Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
        survival Survival (0 = No; 1 = Yes)
        name Name
        sex Sex
        age Age
        sibsp Number of Siblings/Spouses Aboard
        parch Number of Parents/Children Aboard
        ticket Ticket Number
        fare Passenger Fare (British pound)
        cabin Cabin
        embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
        boat Lifeboat
        body Body Identification Number
        home.dest Home/Destination
    """
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn import preprocessing

    """
    # read dataset   
    # change nan to 0 
    # add new column deck, based on first letter in column cabin  
    # test and drop columns which are insignificant or make things worse
    #   the most important info is, whether someone could get a boat (nr>0) or not (nr=0)
    """
    df = pd.read_excel('data/titanic.xls')
    #df_display(df, "0: ", "head", 5)
    df.fillna(0, inplace=True)
    df.insert(11, 'deck', df.apply(lambda row: str(row.cabin)[0], axis=1))
    df.drop(['body', 'name', 'cabin'], axis=1, inplace=True)
    #df_display(df, "1: ", "head&tail", 5)

    """
    # change data in df
    """
    def handle_non_numerical_data(df):
        """
        make changes in columns:
            sex, ticket, cabin, embarked, boat, home.dest
        """
        """
        # get column names and iterate through them
        """
        columns = df.columns.values
        for column in columns:

            """
            # define dict and function used further down
            """
            text_digit_vals = {}

            def convert_to_int(val):
                return text_digit_vals[val]

            """
            # if the column data type is not int64 or float64: make changes
            """
            if df[column].dtype != np.int64 and df[column].dtype != np.float64:

                """
                # get unique elements and iterate through them
                """
                column_contents = df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:

                    """
                    # map each unique value to a number x=0, 1, 2, ... and store it in dict
                    """
                    #print("2.1: ", column, " --- ", unique)
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x += 1

                """
                # map feeds each column value into function convert_to_int, which returns value of dict[key]
                # convert_to_int(val): return text_digit_vals[val]
                """
                df[column] = list(map(convert_to_int, df[column]))
        return df

    """
    # automatic replacement of non numerical data
    """
    df = handle_non_numerical_data(df)
    df_display(df, "3: ", "ht", 5)

    """
    # K-Means-Clustering 
    # Unsupervised
    #   data: X    
    #   label: y survived with 1 (non-survival) or 0 (survival)
    #   found clusters: survival or non-survival group has arbitrarily either 1 or 0 
    # preprocessing.scale(X): 
    #   improves accuracy here by 20 to 30%
    #   it aims to put your data in a range from -1 to +1, which can make things better.
    """
    X = np.array(df.drop(['survived'], axis=1).astype(float))
    X = preprocessing.scale(X)
    y = np.array(df['survived'])
    clf = KMeans(n_clusters=2)
    kmeans=clf.fit(X)
    #print("kmeans.labels: ", kmeans.labels_)
    #print("kmeans.cluster_centers_: ", kmeans.cluster_centers_)

    """
    # accuracy
    """
    correct = 0
    for i in range(len(X)):

        """
        # get a sample and reshape
        # sample from same already preprocessed X, as used for KMeans.fit()
        # .reshape(-1, len()): -1 is used for unknown resulting dimension
        """
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        #print("predict_me: ", predict_me, predict_me.shape)

        """
        # make prediction and compare to label y
        # since prediction label is arbitrary, then an accuracy of constantly 0.7 or 0.3 means 70%
        """
        prediction = clf.predict(predict_me)
        #print("prediction: ", prediction)
        if prediction[0] == y[i]:
            correct += 1
    print("accuracy: ", correct / len(X))


def fifteen_Clustering_KMeans_programedFormulas():
    """
    KMeans clustering
    programed formulas
    """
    """
    # sample data
    """
    X = np.array([[1, 2],
                  [1.5, 1.8],
                  [5, 8],
                  [8, 8],
                  [1, 0.6],
                  [9, 11]])
    plt.scatter(X[:, 0], X[:, 1], marker="o", s=150)
    plt.show()
    print("\n X:\n ", X)

    """
    # KMeans class
    """
    class K_Means(object):
        """
        programed formulas
        """
        """
        # called when K_Means object is created
        """
        def __init__(self, k=2, tol=0.001, max_iter=300):
            self.k = k
            self.tol = tol
            self.max_iter = max_iter

        """
        # fit(): unsupervised clustering of the data
        """
        def fit(self, data):
            """
            determines dicts
                self.centroids
                self.classifications
            """
            """
            # dict that contains the cluster label as key and the cluster centers as value
            #   starts with the first two samples in the dataset and uses them as centers
            #   {0: array([1., 2.]), 1: array([1.5, 1.8])}
            """
            self.centroids = {}
            for i in range(self.k):
                self.centroids[i] = data[i]
            print("\n1: centroids start\n ", self.centroids)

            """
            # optimization
            """
            for i in range(self.max_iter):

                """
                # dict that contains the assumed classification labels as key and the sample features as value
                """
                self.classifications = {}
                for i in range(self.k):
                    self.classifications[i] = []

                """
                # iterate through samples (featureset) in data
                """
                for featureset in data:

                    """
                    # Classificaton
                    # re-classify samples
                    # distances:
                    #   the distance between (or norm of) sample (featureset) and each of the centroids
                    #   [7.211102550927978, 2.1223571801183696]
                    # classification:
                    #   assign sample (featureset) to the cluster, to whose centroid it has the minimal distance
                    # classifications:
                    #   {0: [array([1., 2.])], 1: [array([1.5, 1.8]), array([5., 8.]), array([8., 8.]), array([1. , 0.6]), array([ 9., 11.])]}
                    #   {0: [array([1., 2.]), array([1.5, 1.8]), array([1. , 0.6])], 1: [array([5., 8.]), array([8., 8.]), array([ 9., 11.])]}
                    #   {0: [array([1., 2.]), array([1.5, 1.8]), array([1. , 0.6])], 1: [array([5., 8.]), array([8., 8.]), array([ 9., 11.])]}
                    """
                    distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    #print("\n2.1: distances\n ", distances)
                    #print("\n2.2: classification\n ", classification)
                    self.classifications[classification].append(featureset)
                print("\n2: classifications ----\n ", self.classifications)

                """
                # Centroids
                # prev_centroids: store version of last optimization iteration step
                # re-calculate the center of the samples in each cluster
                #   the cluster centroid is the np.average() of the cluster
                # {0: array([1., 2.]), 1: array([4.9 , 5.88])}
                # {0: array([1.16666667, 1.46666667]), 1: array([7.33333333, 9.        ])}
                # {0: array([1.16666667, 1.46666667]), 1: array([7.33333333, 9.        ])}
                """
                prev_centroids = dict(self.centroids)
                for classification in self.classifications:
                    self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                print("\n3: centroids:\n ", self.centroids)

                """
                # Tolerance
                # check if tolerance is reached
                #
                # 1. example second iteration
                #    prev_centroid:  [1. 2.]
                #    current_centroid:  [1.16666667 1.46666667]
                #    np.sum( ([1.16666667 1.46666667]-[1. 2.]) / [1. 2.] * 100.0)
                #    (1.16666667-1)/1.*100 + (1.46666667-2.)/2.*100 = -9.999999500000012
                #    prev_centroid:  [4.9  5.88]
                #    current_centroid:  [7.33333333 9.        ]
                #    np.sum( ([7.33333333 9.        ]-[4.9  5.88]) / [4.9  5.88] * 100.0)
                #    (7.33333333-4.9)/4.9*100.0 + (9.-5.88)/5.88* 100.0 = 102.72108836734694
                # 2. all np.sum() 
                # 0.0
                # 453.3333333333334
                #
                # -9.999999999999996
                # 102.72108843537411
                # 
                # 0.0
                # 0.0
                """
                optimized = True
                for c in self.centroids:
                    prev_centroid = prev_centroids[c]
                    current_centroid = self.centroids[c]
                    print("\n4.1 prev_centroid: ", prev_centroid)
                    print("\n4.2 current_centroid: ", current_centroid)
                    if np.sum((current_centroid - prev_centroid) / prev_centroid * 100.0) > self.tol:
                        optimized = False
                    print("\n4. centroid change: ", np.sum((current_centroid - prev_centroid) / prev_centroid * 100.0))

                """
                done
                """
                if optimized:
                    break

        """
        # predict(): predicts new data, based on clustering result of fit() with training data
        """
        def predict(self, data):
            distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            return classification

    """
    # use own K_Means class and fit data
    """
    clf = K_Means(k=2, max_iter=200)
    clf.fit(X)

    """
    # plot centroids as black dots
    """
    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                    marker="o", color="k", s=150, linewidths=5)

    """
    # plot data as green dots and red dots
    """
    colors = ["g", "r", "c", "y"]
    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker="o", color=color, s=150, linewidths=5)

    """
    # predictions for new data and plot them as green stars and red stars
    """
    unknowns = np.array([[1, 3],
                         [8, 9],
                         [0, 3],
                         [5, 4],
                         [6, 4], ])
    for unknown in unknowns:
        classification = clf.predict(unknown)
        plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)
    plt.show()


def sixteen_Clustering_KMeans_TitanicData_programedFormulas():
    """
    algorithm of: fifteen_Clustering_KMeans_programedFormulas()
    applied to data of: fourteen_Clustering_KMeans_TitanicData()
    """
    import pandas as pd
    from sklearn import preprocessing

    """
    # read dataset   
    # change nan to 0 
    # add new column deck, based on first letter in column cabin  
    # test and drop columns which are insignificant or make things worse
    #   the most important info is, whether someone could get a boat (nr>0) or not (nr=0)
    """
    df = pd.read_excel('data/titanic.xls')
    # df_display(df, "0: ", "head", 5)
    df.fillna(0, inplace=True)
    df.insert(11, 'deck', df.apply(lambda row: str(row.cabin)[0], axis=1))
    df.drop(['body', 'name', 'cabin', 'home.dest', 'parch', 'sibsp', 'embarked', 'ticket'], axis=1, inplace=True)
    df_display(df, "1: ", "h", 5)

    """
    # change data in df
    """
    def handle_non_numerical_data(df):
        """
        make changes in columns:
            sex, ticket, cabin, embarked, boat, home.dest
        """
        """
        # get column names and iterate through them
        """
        columns = df.columns.values
        for column in columns:

            """
            # define dict and function used further down
            """
            text_digit_vals = {}

            def convert_to_int(val):
                return text_digit_vals[val]

            """
            # if the column data type is not int64 or float64: make changes
            """
            if df[column].dtype != np.int64 and df[column].dtype != np.float64:

                """
                # get unique elements and iterate through them
                """
                column_contents = df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:

                    """
                    # map each unique value to a number x=0, 1, 2, ... and store it in dict
                    """
                    # print("2.1: ", column, " --- ", unique)
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x += 1

                """
                # map feeds each column value into function convert_to_int, which returns value of dict[key]
                # convert_to_int(val): return text_digit_vals[val]
                """
                df[column] = list(map(convert_to_int, df[column]))
        return df

    """
    # automatic replacement of non numerical data
    """
    df = handle_non_numerical_data(df)
    #df_display(df, "3: ", "ht", 5)

    """
    # KMeans class
    """
    class K_Means(object):
        """
        programed formulas
        """
        """
        # called when K_Means object is created
        """

        def __init__(self, k=2, tol=0.001, max_iter=300):
            self.k = k
            self.tol = tol
            self.max_iter = max_iter

        """
        # fit(): unsupervised clustering of the data
        """
        def fit(self, data):
            """
            determines dicts
                self.centroids
                self.classifications
            """
            """
            # dict that contains the cluster label as key and the cluster centers as value
            #   starts with the first two samples in the dataset and uses them as centers
            #   {0: array([1., 2.]), 1: array([1.5, 1.8])}
            """
            self.centroids = {}
            for i in range(self.k):
                self.centroids[i] = data[i]
            print("\n1: centroids start\n ", self.centroids)

            """
            # optimization
            """
            for iter in range(self.max_iter):
                print("\ni: ", iter, "\nmax_iter: ", self.max_iter)

                """
                # dict that contains the assumed classification labels as key and the sample features as value
                """
                self.classifications = {}
                for i in range(self.k):
                    self.classifications[i] = []

                """
                # iterate through samples (featureset) in data
                """
                for featureset in data:
                    """
                    # Classificaton
                    # re-classify samples
                    # distances:
                    #   the distance between (or norm of) sample (featureset) and each of the centroids
                    #   [7.211102550927978, 2.1223571801183696]
                    # classification:
                    #   assign sample (featureset) to the cluster, to whose centroid it has the minimal distance
                    # classifications:
                    #   {0: [array([1., 2.])], 1: [array([1.5, 1.8]), array([5., 8.]), array([8., 8.]), array([1. , 0.6]), array([ 9., 11.])]}
                    #   {0: [array([1., 2.]), array([1.5, 1.8]), array([1. , 0.6])], 1: [array([5., 8.]), array([8., 8.]), array([ 9., 11.])]}
                    #   {0: [array([1., 2.]), array([1.5, 1.8]), array([1. , 0.6])], 1: [array([5., 8.]), array([8., 8.]), array([ 9., 11.])]}
                    """
                    distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    # print("\n2.1: distances\n ", distances)
                    # print("\n2.2: classification\n ", classification)
                    self.classifications[classification].append(featureset)
                #print("\n2: classifications ----\n ", self.classifications)

                """
                # Centroids
                # prev_centroids: store version of last optimization iteration step
                # re-calculate the center of the samples in each cluster
                #   the cluster centroid is the np.average() of the cluster
                # {0: array([1., 2.]), 1: array([4.9 , 5.88])}
                # {0: array([1.16666667, 1.46666667]), 1: array([7.33333333, 9.        ])}
                # {0: array([1.16666667, 1.46666667]), 1: array([7.33333333, 9.        ])}
                """
                prev_centroids = dict(self.centroids)
                for classification in self.classifications:
                    self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                print("\n3: centroids:\n ", self.centroids)

                """
                # Tolerance
                # check if tolerance is reached
                #
                # 1. example second iteration
                #    prev_centroid:  [1. 2.]
                #    current_centroid:  [1.16666667 1.46666667]
                #    np.sum( ([1.16666667 1.46666667]-[1. 2.]) / [1. 2.] * 100.0)
                #    (1.16666667-1)/1.*100 + (1.46666667-2.)/2.*100 = -9.999999500000012
                #    prev_centroid:  [4.9  5.88]
                #    current_centroid:  [7.33333333 9.        ]
                #    np.sum( ([7.33333333 9.        ]-[4.9  5.88]) / [4.9  5.88] * 100.0)
                #    (7.33333333-4.9)/4.9*100.0 + (9.-5.88)/5.88* 100.0 = 102.72108836734694
                # 2. all np.sum() 
                # 0.0
                # 453.3333333333334
                #
                # -9.999999999999996
                # 102.72108843537411
                # 
                # 0.0
                # 0.0
                """
                optimized = True
                for c in self.centroids:
                    prev_centroid = prev_centroids[c]
                    current_centroid = self.centroids[c]
                    print("\n4.1 prev_centroid: ", prev_centroid)
                    print("\n4.2 current_centroid: ", current_centroid)
                    if np.abs(np.sum((current_centroid - prev_centroid) / prev_centroid * 100.0)) > self.tol:
                        optimized = False
                    print("\n4. centroid change: ", np.sum((current_centroid - prev_centroid) / prev_centroid * 100.0))

                """
                done
                """
                if optimized:
                    print("k-means done, iterations: ", iter)
                    break

        """
        # predict(): predicts new data, based on clustering result of fit() with training data
        """
        def predict(self, data):
            distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            return classification

    """
    # K-Means-Clustering 
    # Unsupervised
    #   data: X    
    #   label: y survived with 1 (non-survival) or 0 (survival)
    #   found clusters: survival or non-survival group has arbitrarily either 1 or 0 
    # preprocessing.scale(X): 
    #   improves accuracy here by 20 to 30%
    #   it aims to put your data in a range from -1 to +1, which can make things better.
    """
    X = np.array(df.drop(['survived'], axis=1).astype(float))
    X = preprocessing.scale(X)
    y = np.array(df['survived'])
    cols=df.columns.drop('survived')
    print(cols)
    df_display(pd.DataFrame(X, columns=cols), "0: ", "ht", 5)
    clf = K_Means(k=2)
    clf.fit(X)

    """
    # predictions for new data and plot them as green stars and red stars
    """
    """
    # accuracy
    """
    correct = 0
    for i in range(len(X)):

        """
        # get a sample and reshape
        # sample from same already preprocessed X, as used for KMeans.fit()
        # .reshape(-1, len()): -1 is used for unknown resulting dimension; change from [] to [[]]
        """
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        # print("predict_me: ", predict_me, predict_me.shape)

        """
        # make prediction and compare to label y
        # since prediction label is arbitrary, then an accuracy of constantly 0.7 or 0.3 means 70%
        """
        prediction = clf.predict(predict_me)
        # print("prediction: ", prediction)
        if prediction == y[i]:
            correct += 1
    print("accuracy: ", correct / len(X))


def seventeen_hierarchicalClustering_meanShift():
    """
    Mean Shift is very similar to the K-Means algorithm, except for one very important factor:
    you do not need to specify the number of groups prior to training.
    The Mean Shift algorithm finds clusters on its own.
    For this reason, it is even more of an "unsupervised" machine learning algorithm than K-Means.
    What is Mean Shift used for?
    Along with the clustering, Mean Shift is also very popular in image analysis
    for both tracking and smoothing.
    """
    from sklearn.cluster import MeanShift
    from sklearn.datasets import make_blobs
    from mpl_toolkits.mplot3d import Axes3D

    """
    # data generation: 100 samples around 3 centers
    """
    centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
    X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1.5)
    print("cluster centers: \n", centers)

    """
    # fit
    """
    ms = MeanShift()
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    """
    # info
    """
    print("\nestimated cluster centers: \n", cluster_centers)
    n_clusters_ = len(np.unique(labels))
    print("\nNumber of estimated clusters:", n_clusters_)

    """
    # visualize
    # matplotlib 3d subplot: samples X as dots, cluster centers as black x 
    """
    colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y', 'm']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(X)):
        ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
               marker="x", color='k', s=150, linewidths=5, zorder=10)
    plt.show()


def eighteen_hierarchicalClustering_meanShift_TitanicData():
    """
    with titanic data:
    Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
    survival Survival (0 = No; 1 = Yes)
    name Name
    sex Sex
    age Age
    sibsp Number of Siblings/Spouses Aboard
    parch Number of Parents/Children Aboard
    ticket Ticket Number
    fare Passenger Fare (British pound)
    cabin Cabin
    embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
    boat Lifeboat
    body Body Identification Number
    home.dest Home/Destination
    """
    from sklearn.cluster import MeanShift
    from sklearn import preprocessing
    import pandas as pd

    """
    # load data
    """
    df = pd.read_excel('data/titanic.xls')
    original_df = pd.DataFrame.copy(df)

    """
    # non numerical data to numerical data
    """
    def handle_non_numerical_data(df):
        columns = df.columns.values
        for column in columns:
            text_digit_vals = {}
            def convert_to_int(val):
                return text_digit_vals[val]
            # print(column,df[column].dtype)
            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                column_contents = df[column].values.tolist()
                # finding just the uniques
                unique_elements = set(column_contents)
                # great, found them.
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        # creating dict that contains new
                        # id per unique string
                        text_digit_vals[unique] = x
                        x += 1
                # now we map the new "id" vlaue
                # to replace the string.
                df[column] = list(map(convert_to_int, df[column]))
        return df

    """
    # drop some columns 
    # apply the conversion from non-numerical to numerical
    """
    df.drop(['body', 'name'], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df.drop(['ticket', 'home.dest'], axis=1, inplace=True)
    df = handle_non_numerical_data(df)

    """
    # preprocessing and fitting
    """
    X = np.array(df.drop(['survived'], axis=1).astype(float))
    X = preprocessing.scale(X)
    y = np.array(df['survived'])
    clf = MeanShift()
    clf.fit(X)

    """
    # new columnn cluster_group: add found labels to samples
    """
    labels = clf.labels_
    cluster_centers = clf.cluster_centers_
    original_df['cluster_group'] = np.nan
    for i in range(len(X)):
        original_df['cluster_group'].iloc[i] = labels[i]
    df_display(original_df, "1: ", "h", 10)

    """
    # survival rates of each found cluster
    """
    n_clusters_ = len(np.unique(labels))
    survival_rates = {}
    for i in range(n_clusters_):
        """
        # only samples of cluster i
        """
        temp_df = original_df[(original_df['cluster_group'] == float(i))]
        """
        # only samples of cluster i who survived
        """
        survival_cluster = temp_df[(temp_df['survived'] == 1)]
        """
        # survival rate within cluster i
        """
        survival_rate = len(survival_cluster) / len(temp_df)
        survival_rates[i] = survival_rate
    print(survival_rates)

    """
    # further digging into dataframes
    """
    # one filter on column
    df_display(original_df[ (original_df['cluster_group'] == 2) ])
    # one filter on column and .describe(): shows statistics
    print(original_df[ (original_df['cluster_group'] == 0) ].describe())
    # two filters on columns: .isna() filters for np.nan; only show age column; only show .shape
    print(original_df[ (original_df['cluster_group'] == 0) & (original_df['age'].isna()) ]['age'].shape)
    # two filters on columns
    print(original_df[ (original_df['cluster_group'] == 3) & (original_df['survived'] == 1) ])
    # .query(): three filters on column
    print(original_df.query( 'cluster_group==1 & survived==1 & age<10' ))
    # three filters on columns: .isin([,,,]) for filter list
    print(original_df[ (original_df.cluster_group != 0) & (original_df.survived == 1) & (original_df.age.isin([29, 30, 31, 32])) ])
    # .loc[]: the same as []
    print(original_df.loc[ (original_df.cluster_group != 0) & (original_df.survived == 1) & (original_df.age.isin([29, 30, 31, 32])) ])
    # pd.DataFrame(): constructs new df
    df_new = pd.DataFrame( df[ original_df.cluster_group != 0 ] )
    df_display(df_new, "df_new: ", "h")
    # df.assign(column_name=lambda row ): assign to column_name;
    # use .astype('string') or .astype('float'); str() or float() applied to whole column
    print(df.assign( cabin_fare=lambda row: row.cabin.astype('string') + " @ " + row.fare.astype('string') ))
    print(df.assign( age_boat=lambda row: row.age.astype('float') + row.boat.astype('float') ))
    # apply(lambda row: row.name, axis=1): row.name is the index
    print(df.apply( lambda row: row.age.astype('float') + row.boat.astype('float') if row.name <= 2 else 0, axis=1 ))
    # new df_1 as copy of df and adapted fare column
    df_1 = pd.DataFrame(df)
    df_1['fare'] = df['fare'].apply( lambda row: row - 2 )
    print(df_1)
    # new df_1 only with the adapted fare column
    df_1_ = pd.DataFrame()
    df_1_['fare'] = df['fare'].apply( lambda row: row - 2 )
    print(df_1_)
    # new df_2 Series of index and one column
    df_2 = df.apply( lambda row: row.age.astype('float') + row.boat.astype('float') if row.name <= 2 else 0, axis=1 )
    print(df_2, type(df_2))
    # new df_3 as copy of df and new column cabin_fare
    df_3 = df.assign( cabin_fare=lambda row: row.cabin.astype('string') + " @ " + row.fare.astype('string') )
    print(df_3)
    # new df_4 as copy of df and adapted column fare
    df_4 = pd.DataFrame(df)
    df_4['fare'] = df.apply( lambda row: row.age.astype('float') + row.boat.astype('float') if row.name <= 2 else 0, axis=1 )
    print(df_4)
    # iterate through rows
    for index, row in df.iterrows():
        if index < 10:
            print(index, row["survived"], row["age"])
        else:
            break
    # iterate through columns
    for columnname, values in df.iloc[:5, 0:2].iteritems():
        print(values)
    # iterate through rows, return tuples containing values of one row
    for i in df.iloc[:5, 0:2].itertuples():
        print(i)


def nineteen_hierarchicalClustering_meanShift_programedFormulas():
    """
    MeanShift
    programed formulas
    Remarks:
        It doesn't always work.
        Sometimes it's because the clusters are pretty close.
        Other times I have really no idea when seeing the output, when it appears the output has obvious clusters.
        Maybe we're removing too many "duplicate" clusters and winding up with too few.
        No idea! Regardless, I am giving myself a passing grade and we're moving on...
    """
    from functools import reduce
    from sklearn.datasets import make_blobs

    """
    # generate samples
    """
    if 0:
        X = np.array([[1, 2],
                      [1.5, 1.8],
                      [5, 8],
                      [8, 8],
                      [1, 0.6],
                      [9, 11],
                      [8, 2],
                      [10, 2],
                      [9, 3], ])

        #plt.scatter(X[:,0], X[:,1], s=150)
        #plt.show()
    else:
        X, y = make_blobs(n_samples=15, centers=3, n_features=2)

    """
    # Mean Shift class
    """
    class Mean_Shift(object):
        """
        Mean Shift algorithm
        """
        """
        # called when object is generated
        # store the radius used in fit() to update centroids
        # updated version: 
        #   the plan here is to create a massive radius, but make that radius go in steps, like bandwidths, 
        #   or a bunch of radiuses with different lengths, which we'll call steps. 
        #   If a featureset is in the closest radius, it will have a much higher "weight" than one much further away.
        """
        # def __init__(self, radius=4):
        #     self.radius = radius
        def __init__(self, radius=None, radius_norm_step=100):
            self.radius = radius
            self.radius_norm_step = radius_norm_step

        """
        # fit
        """
        def fit(self, data):
            """
            fit dataset
            """
            """
            # set radius
            # updated version:
            #   if the user hasn't hard-coded the radius, (self.radius==None)
            #   then we're going to find the one "center" of ALL of the data. (np.average)
            #   Then, we will take the norm of that data, 
            #   then we say each radius with self.radius is basically the full data-length, 
            #   divided by how many steps we want it to have.
            #   all_data_centroid:
            #       [5.83333333 4.26666667]     
            #   all_data_norm 
            #       7.22718632817933
            #   self.radius:
            #       0.0722718632817933
            """
            if self.radius == None:
                all_data_centroid = np.average(data, axis=0)
                all_data_norm = np.linalg.norm(all_data_centroid)
                self.radius = all_data_norm / self.radius_norm_step
                #print("\nall_data_centroid:\n", all_data_centroid)
                #print("all_data_norm", all_data_norm)
                #print("\nself.radius:\n", self.radius)

            """
            # start: assign each sample to a centroids key i
            # len(data): 9
            # centroids:
            #  {0: array([1., 2.]), 1: array([1.5, 1.8]), 2: array([5., 8.]), 3: array([8., 8.]), 4: array([1. , 0.6]), 
            #   5: array([ 9., 11.]), 6: array([8., 2.]), 7: array([10.,  2.]), 8: array([9., 3.])}
            """
            centroids = {}
            for i in range(len(data)):
                centroids[i] = data[i]
            #print("len data", len(data))
            #print("\ncentroids:\n", centroids)

            """
            # optimization loop
            # iter: no purpose, just to print() 
            # updated version:
            #   weights: range(100)][::-1] is the same as range(99, -1,-1), 
            #            i.e. i for i in range(100)[::-1] iterates through 99,98,97...1,0
            #            and [i for i in range(99)][::-1] generates list [99,98,97...1,0]
            """
            iter = 0
            weights = [i for i in range(self.radius_norm_step)][::-1]
            while True:
                iter+=1
                """
                # iterate through centroids to update them
                """
                new_centroids = []
                for i in centroids:
                    in_bandwidth = []
                    centroid = centroids[i]
                    """
                    # check each sample with centroid: if norm smaller than cluster radius append to in_bandwidth
                    # new centroid: the average of those samples in_bandwidth; append it to new_centroids as tuple
                    #   e.g. list:  [1.16666667 1.46666667]; tuple: (1.1666666666666667, 1.4666666666666666)
                    # updated version:
                    #   new way to calculate in_bandwith
                    #       calculate distance (norm) of sample (featureset) from centroid
                    #       avoid division by 0
                    #       weight_index: within radius int(0.9) becomes 0
                    #                     outside the radius_norm_step which is 100, set to 100-1
                    #       (weights[weight_index] ** 2): (weights[0] ** 2) becomes 99^2 or 9801
                    #       (weights[weight_index] ** 2) * [[5,8]] becomes [[5,8][5,8]...[5,8]] of length 9801
                    #           where [featureset] is actually of the form [array([5., 8.])]
                    #           length of in_bandwith is e.g. 49348
                    #   new_centroids
                    #     at the first iteration, iter 1, there are 9 centroids, one for each sample
                    #     at each iteration, the algorithm loads one centroid after another 
                            updates it with the data and adds it to new_centroids 
                    #     at the last iteration, iter 14, only three centroids remain
                    #       [(1.182114380539928, 1.501096855141567)]
                    #       [(1.182114380539928, 1.501096855141567), (7.455989166564077, 8.442632032500308)]
                    #       [(1.182114380539928, 1.501096855141567), (7.455989166564077, 8.442632032500308), (8.97302521369357, 2.5055195223572873)]
                    """
                    for featureset in data:
                        # if np.linalg.norm(featureset-centroid) < self.radius:
                        #    in_bandwidth.append(featureset)
                        distance = np.linalg.norm(featureset - centroid)
                        if distance == 0:
                            distance = 0.00000000001
                        weight_index = int(distance / self.radius)
                        if weight_index > self.radius_norm_step - 1:
                            weight_index = self.radius_norm_step - 1
                        to_add = (weights[weight_index] ** 2) * [featureset]
                        #print("\n[featureset]\n" ,[featureset])
                        #print("\nto_add: \n", to_add)
                        in_bandwidth += to_add
                    new_centroid = np.average(in_bandwidth, axis=0)
                    new_centroids.append(tuple(new_centroid))
                    #print("\nin_bandwidth length\n", reduce(lambda count, l: count + len(l), in_bandwidth, 0))
                    #print("1\n",new_centroid)
                    #print("2\n", tuple(new_centroid))
                    #print("\niter\n", iter)
                    #print("\nnew_centroids\n", new_centroids)

                """
                # store centroids of last iteration before updating them
                # assign each unique centroid to dict centroids
                # iter:
                #  1
                #  2
                #  3
                # new_centroids:
                #  [(1.1666666666666667, 1.4666666666666666), (1.1666666666666667, 1.4666666666666666), (6.5, 8.0), 
                #   (7.333333333333333, 9.0), (1.1666666666666667, 1.4666666666666666), (8.5, 9.5), (9.0, 2.3333333333333335), 
                #   (9.0, 2.3333333333333335), (9.0, 2.3333333333333335)]
                #  [(1.1666666666666667, 1.4666666666666666), (7.333333333333333, 9.0), (7.333333333333333, 9.0), 
                #   (7.333333333333333, 9.0), (9.0, 2.3333333333333335)]
                #  [(1.1666666666666667, 1.4666666666666666), (7.333333333333333, 9.0), (9.0, 2.3333333333333335)]
                # len(uniques):
                #  5
                #  3
                #  3
                # centroids:
                #  {0: array([1.16666667, 1.46666667]), 1: array([6.5, 8. ]), 2: array([7.33333333, 9.        ]), 
                #   3: array([8.5, 9.5]), 4: array([9.        , 2.33333333])}
                #  {0: array([1.16666667, 1.46666667]), 1: array([7.33333333, 9.        ]), 2: array([9.        , 2.33333333])}
                #  {0: array([1.16666667, 1.46666667]), 1: array([7.33333333, 9.        ]), 2: array([9.        , 2.33333333])}
                # 
                # updated version:
                #   before, all we had to do to note convergence was remove the centroids that were identical to eachother. 
                #   With this method, however, it is highly likely that we have centroids that are extremely close, 
                #   but not identical. We want to merge these too.
                #   to_pop:
                #       speeds up algorithm  
                #       used to store those centroids we also want to have removed from uniques
                #           because they are near each other, i.e. smaller than self.radius
                """
                prev_centroids = dict(centroids)
                centroids = {}
                uniques = sorted(list(set(new_centroids)))
                to_pop = []
                for i in uniques:
                    for ii in [i for i in uniques]:
                        if i == ii:
                            pass
                        elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                            # print(np.array(i), np.array(ii))
                            to_pop.append(ii)
                            break
                for i in to_pop:
                    try:
                        uniques.remove(i)
                    except:
                        pass
                for i in range(len(uniques)):
                    centroids[i] = np.array(uniques[i])
                #print("iter", iter)
                #print("\nnew_centroids:\n", new_centroids)
                #print("len data", len(uniques))
                #print("\ncentroids:\n", centroids)

                """
                # check if no centroid has changed since last iteration 
                """
                optimized = True
                for i in centroids:
                    if not np.array_equal(centroids[i], prev_centroids[i]):
                        optimized = False
                    if not optimized:
                        break
                """
                # done
                """
                if optimized:
                    break
            """
            # save
            """
            self.centroids = centroids

            """
            # updated version
            #   also classify the existing featuresets
            #   All this does is take known featuresets, and calculate the minimum distance to the centroids, 
            #   and classify as belonging to the closest centroid.
            """
            self.classifications = {}
            for i in range(len(self.centroids)):
                self.classifications[i] = []

            for featureset in data:
                # compare distance to either centroid
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                # print(distances)
                classification = (distances.index(min(distances)))

                # featureset that belongs to that cluster
                self.classifications[classification].append(featureset)

        """
        # predict
        """
        def predict(self, data):
            # compare distance to either centroid
            distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
            classification = (distances.index(min(distances)))
            return classification

    """
    # fit samples
    """
    clf = Mean_Shift()
    clf.fit(X)

    """
    # plot samples and found centroids
    """
    plt.scatter(X[:, 0], X[:, 1], s=150)
    centroids = clf.centroids
    colors = 10 * ["g", "r", "c", "b", "k"]
    for c in centroids:
        plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
    plt.show()


def df_display(df, txt="", head_tail="", n=5 ):
    print('\n', txt, '\n')
    if head_tail == "head" or head_tail == "h":
        print(tabulate(df.head(n), headers='keys', tablefmt='psql', showindex=True))
    elif head_tail == "tail" or head_tail == "t":
        print(tabulate(df.tail(n), headers='keys', tablefmt='psql', showindex=True))
    elif head_tail == "head&tail" or head_tail == "ht":
        print(tabulate(df.head(n), headers='keys', tablefmt='psql', showindex=True))
        print(tabulate(df.tail(n), headers='keys', tablefmt='psql', showindex=True))
    else:
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=True))
    print('\n', df.shape, '\n\n')


if __name__ == "__main__":
    if 0:
        one_k_means_clustering()
        two_mean_shift_clustering_2D()
        two_mean_shift_clustering_3D()
        three_regression()
        four_regression_forecasts()
        five_pickle_loadTrainedClassifiers()
        six_linearRegression_programedFormulas()
        seven_creatingSampleData_forTesting()
        eight_applying_KNearestNeighbors_toData()
        nine_KNearestNeighbors_programedFormulas()
        ten_supportVectorMachine()
        eleven_SupportVectorMachine_programedFormulas()
        twelve_kernels_softMarginSVM_CVXOPT("test_linear")
        twelve_kernels_softMarginSVM_CVXOPT("test_non_linear")
        twelve_kernels_softMarginSVM_CVXOPT("test_non_linear_2")
        twelve_kernels_softMarginSVM_CVXOPT("test_soft")
        thirteen_Clustering_KMeans()
        fourteen_Clustering_KMeans_TitanicData()
        fifteen_Clustering_KMeans_programedFormulas()
        sixteen_Clustering_KMeans_TitanicData_programedFormulas()
        seventeen_hierarchicalClustering_meanShift()
        eighteen_hierarchicalClustering_meanShift_TitanicData()
        nineteen_hierarchicalClustering_meanShift_programedFormulas()
    else:
        nineteen_hierarchicalClustering_meanShift_programedFormulas()


