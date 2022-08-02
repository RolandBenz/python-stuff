import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
from contextlib import contextmanager
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm, preprocessing


# https://pandas.pydata.org/docs/reference/index.html
# https://pandas.pydata.org/docs/user_guide/index.html
# https://www.kaggle.com/

"""
Terminal:
Tut_Pandas> pip install jupyterlab
Tut_Pandas> jupyter lab
Browser:
url>  http://localhost:8888/lab?token=8cc5888a814ddf00c9546a258b27f9c9e512540fd0c1f4c1
Jupiter Lab Browser App:
file > new > notebook >Python 3
file > save notebook as... > Tut_Pandas> Tutorial-pandas.ipynb
"""


def one_filterForRegion_setDateAsIndex():
    # 1.1 load data into data frame
    # ******************
    df = pd.read_csv("avocado.csv")
    # ******************
    # print only the first x records
    print("\n1.1:\n")
    print(df.head(2), df.tail(2))
    # 1.2 print column AveragePrice; head default at 5 records
    #  attribute-like dot notation is possible, but rarely used
    print("\n1.2:\n")
    print(df['AveragePrice'].head(), df.AveragePrice.head())
    # 1.3 filter for column region where value is Albany
    # ******************
    albany_df = df[df['region'] == "Albany"]
    # ******************
    print("\n1.3:\n")
    print(albany_df.head())
    # 1.4 df are often indexed by something
    #  in this case it is just incrementing row counts, which is useless.
    #  the data is organized by date, so we index it for this column
    print("\n1.4:\n")
    print(albany_df.index)
    # Some methods in pandas will modify your dataframe in place,
    # but MOST are going to simply do the thing and return a new dataframe.
    albany_df.set_index("Date")
    print(albany_df.head())
    # One way is to just define a variable or re-define the existing one
    # ******************
    albany_df = albany_df.set_index("Date")
    # ******************
    print(albany_df.head())
    # 1.5 When we call .plot() on a dataframe, it is just assumed that the x-axis will be your index,
    # and then Y will be all of your columns, which is why we specified one column in particular.
    albany_df['AveragePrice'].plot()
    plt.show()
    # ******************
    albany_df.plot(y='AveragePrice')  # x = index
    plt.show()
    albany_df.plot(x='year', y='AveragePrice', kind='scatter')
    plt.show()
    # ******************
    sns.displot(albany_df['AveragePrice'])
    plt.show()


def two_convertToDatetime_sortDate_rollingMean_toMarkdown():
    # 2.1 load data into data frame
    df = pd.read_csv("avocado.csv")
    # 2.2 Pandas comes built in with ways to handle for dates.
    # First, we need to convert the date column to datetime objects:
    # ******************
    df['Date'] = pd.to_datetime(df['Date'])
    # ******************
    # 2.3 filter for column region where value is Albany
    albany_df = df[df['region'] == "Albany"]
    # 2.4 index df for column Date
    albany_df.set_index("Date", inplace=True)
    # 2.5 print everything as markdown and plot it
    # there seems to be two average prices for each date in the df
    print("\n2.1:\n")
    print(albany_df['AveragePrice'].to_markdown())
    albany_df["AveragePrice"].plot()
    plt.show()
    # 2.6 sort df and print it again
    # and indeed there are two records for each date and as it turns out
    # there is a column type with entry organic and conventional
    albany_df.sort_index(inplace=True)
    print(albany_df['AveragePrice'].to_markdown())
    # 2.7 let's add a rolling mean - just to have a nice graph
    albany_df["AveragePrice"].rolling(25).mean().plot()
    plt.show()
    # 2.8 we can also store it in a new column of the df
    print("\n2.2:\n")
    albany_df["price25ma"] = albany_df["AveragePrice"].rolling(25).mean()
    print(albany_df.tail())
    # 2.9 the warning in the console is telling us, that we are making changes in slice of a view not a copy
    # so let's make a copy
    print("\n2.3:\n")
    # ******************
    albany_df = df.copy()[df['region'] == "Albany"]
    albany_df.set_index('Date', inplace=True)
    albany_df.sort_index(inplace=True)
    albany_df["price25ma"] = albany_df["AveragePrice"].rolling(25).mean()
    # ******************
    print(albany_df.head())
    print(albany_df.tail())
    # 2.10 let's take a look at the column region
    print("\n2.4:\n")
    df_reg_val = df['region'].values
    df_reg_val_to_list = df['region'].values.tolist()
    # ******************
    df_reg_val_to_list_set_sorted = sorted(set(df['region'].values.tolist()))
    # ******************
    print(df_reg_val_to_list_set_sorted)
    # ******************
    def_reg_unique = df['region'].unique()
    # ******************
    print(def_reg_unique)


def three_filterForType_rollingMeanPerRegion():
    # 3.1 now we start over again by choosing a type
    print("\n3.1:\n")
    # ******************
    df = pd.read_csv("avocado.csv")
    df = df.copy()[df['type'] == 'organic']
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by="Date", ascending=True, inplace=True)
    print(df.head())
    albany_df = df.copy()[df['region'] == "Albany"]
    albany_df.set_index('Date', inplace=True)
    albany_df.sort_index(inplace=True)
    # ******************
    albany_df['AveragePrice'].plot()
    plt.show()
    # 3.2 create a df which has for each region a column with moving average
    print("\n3.2:\n")
    # now good implementation
    # ******************
    graph_df = pd.DataFrame()
    # iterate through unique names in region
    for region in df['region'].unique():
        # filter: df copy of that region; with index: date; and then sorted
        region_df = df.copy()[df['region'] == region]
        region_df.set_index('Date', inplace=True)
        region_df.sort_index(inplace=True)
        # new columns for each region with rolling mean price
        region_df[f"{region}_price25ma"] = region_df["AveragePrice"].rolling(25).mean()
        # store results of new column in df graph_df
        if graph_df.empty:
            graph_df = region_df[[f"{region}_price25ma"]]  # note the double square brackets! (so df rather than series)
        else:
            graph_df = graph_df.join(region_df[f"{region}_price25ma"])
    # ******************
    print(graph_df.tail())
    graph_df.plot(figsize=(8, 5), legend=False)
    plt.show()


def four_groupBy_setYearAsIndex_removeZeros_statistics():
    # 4.1 dataset is encoded in latin
    # load it, store the csv with utf-8 encoding and load it again
    print("\n4.1:\n")
    # ******************
    if 0:
        df = pd.read_csv("Minimum Wage Data.csv" , encoding="latin")
        df.to_csv("minwage.csv", encoding="utf-8")
    df = pd.read_csv("minwage.csv")
    # ******************
    max_rows=5#None
    max_cols=None
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols, ):
        # Year 1968-2020=53 and State Alabama-Virginia=54 multiplies to 2862 rows
        print(df[['Year', 'State', 'State.Minimum.Wage']])
    # 4.2 group by State
    # this is just some name giving for filtering and no aggregation involved.
    # aggregation is done with .pivotTable()
    print("\n4.2.1:\n")
    df_gb = df.copy().groupby("State")
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols, ):
        print(df_gb[['State', 'Year',  'State.Minimum.Wage']].head(54))
    print("\n4.2.2:\n")
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols, ):
        print(df_gb[['Year', 'State', 'State.Minimum.Wage']].get_group("Alabama").set_index("Year"))
    # 4.3
    print("\n4.3:\n")
    # ******************
    act_min_wage = pd.DataFrame()
    # name: group name string; group: df of one group
    for name, group in df.groupby("State"):
        if act_min_wage.empty:
            with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols, ):
                print("\n4.3.1:\nname: ", name, "\ngroup:\n", group.head().to_markdown(), "\n")
            # the group data for one state:
            # set index Year, filter for one column State.Minimum.Wage, rename the column to state name
            act_min_wage = group.set_index("Year")[["State.Minimum.Wage"]].rename(columns={"State.Minimum.Wage": name})
        else:
            act_min_wage = act_min_wage.join(group.set_index("Year")[["State.Minimum.Wage"]].
                                             rename(columns={"State.Minimum.Wage": name}))
    # ******************
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols, ):
        print("\n4.3.2:\n", act_min_wage.head().to_markdown())
    # 4.4 statistics
    print("\n4.4:\n")
    # ******************
    print("\n4.4.1, describe:\n", act_min_wage.describe(), "\n\n")
    print("\n4.4.2, corr:\n", act_min_wage.corr().head())
    # ******************
    # 4.5 issue zeros, states with minimum wage zero,
    # replace zero with NaN and remove records with NaN, build correlation matrix
    print("\n4.5:\n")
    # ******************
    issue_df = df[df["State.Minimum.Wage"] == 0]
    min_wage_corr = act_min_wage.replace(0, np.NaN).dropna(axis=1).corr()
    # ******************
    print(issue_df['State'].unique())
    print(min_wage_corr)
    # check if problem_state string name is in min_wage_corr column string name
    for problem_state in issue_df['State'].unique():
        if problem_state in min_wage_corr.columns:
            print("Missing something here....")
    # 4.6 check if all records are zero for states with found zeros
    print("\n4.6:\n")
    # ******************
    grouped_issues = issue_df.groupby("State")
    # ******************
    print(grouped_issues.get_group("Alabama").head(3))
    print(grouped_issues.get_group("Alabama")['State.Minimum.Wage'].sum())
    # ******************
    for state, data in grouped_issues:
        if data['State.Minimum.Wage'].sum() != 0.0:
            print("Some data found for", state)
    # ******************


def five_visualizeCorrelations_getLabelsFromWebsiteTables():
    # 5.1 load data from four()
    print("\n5.1:\n")
    df = pd.read_csv("minwage.csv")
    max_rows = 5  # None
    max_cols = None
    act_min_wage = pd.DataFrame()
    for name, group in df.groupby("State"):
        if act_min_wage.empty:
            with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols, ):
                print("\n5.1.1:\nname, group: ", name, "\n", group.to_markdown(), "\n")
            # the group data for one state:
            # set index Year, filter for one column State.Minimum.Wage, rename the column to state name
            act_min_wage = group.set_index("Year")[["State.Minimum.Wage"]].rename(columns={"State.Minimum.Wage": name})
        else:
            act_min_wage = act_min_wage.join(group.set_index("Year")[["State.Minimum.Wage"]].
                                             rename(columns={"State.Minimum.Wage": name}))
    min_wage_corr = act_min_wage.replace(0, np.NaN).dropna(axis=1).corr()
    # 5.2 matshow plot
    print("\n5.2:\n")
    # ******************
    plt.matshow(min_wage_corr)
    plt.show()
    # ******************
    # 5.3 improve matshow plot, corr around 1 green; corr around 0 light red; corr around -1 dark red
    print("\n5.3:\n")
    labels = [c[:2] for c in min_wage_corr.columns]  # create buggy abbv state names.
    fig = plt.figure(figsize=(12, 12))  # figure so we can add axis
    ax = fig.add_subplot(111)  # define axis, so we can modify
    ax.matshow(min_wage_corr, cmap=plt.cm.RdYlGn)  # display the matrix
    ax.set_xticks(np.arange(len(labels)))  # show them all!
    ax.set_yticks(np.arange(len(labels)))  # show them all!
    ax.set_xticklabels(labels)  # set to be the abbv (vs useless #)
    ax.set_yticklabels(labels)  # set to be the abbv (vs useless #)
    plt.show()
    # 5.4 searches and imports tables from websites via requests and pandas' read_html()
    print("\n5.4:\n")
    # ******************
    if 0:
        web = requests.get("https://www.infoplease.com/state-abbreviations-and-state-postal-codes")
        dfs = pd.read_html(web.text)
        print(dfs)
    # ******************
    # 5.5 searches and imports tables from websites directly with pandas' read_html()
    print("\n5.5:\n")
    # ******************
    dfs = pd.read_html("https://www.infoplease.com/state-abbreviations-and-state-postal-codes")
    df_state_abbv = dfs[0]
    # ******************
    print(dfs, "\n\n")
    print(df_state_abbv)
    # 5.6 save in case they stop allowing robot access, and load it again
    print("\n5.6:\n")
    if 0:
        df_state_abbv.to_csv("state_abbv.csv")
    state_abbv = pd.read_csv("state_abbv.csv")
    print(state_abbv.head())
    # 5.7 save it again without index and load it again
    print("\n5.7:\n")
    # ******************
    if 0:
        df_state_abbv[["State Name/District", "Postal Code"]]. \
            to_csv("state_abbv.csv", index=False)  # index in this case is worthless, save without
    state_abbv = pd.read_csv("state_abbv.csv", index_col=0)  # load saying the first column is the index column
    # ******************
    print(state_abbv)
    print("index: ", state_abbv.index.name)
    # 5.8 dictionary and from that the abbreviations
    print("\n5.8:\n")
    # ******************
    # without .copy(): abbv_dict becomes a view not a copy of abbv_dict_: after adding Guam, it shows in both.
    # source: https://www.practicaldatascience.org/html/views_and_copies_in_pandas.html
    #         Which leads me to what I will admit is an infuriating piece of advice to have to offer:
    #         if you take a subset for any purpose other than immediately analyzing, you should add .copy() to that subsetting.
    #         Seriously. Just when in doubt, .copy().
    abbv_dict_ = state_abbv.to_dict().copy()
    abbv_dict = abbv_dict_['Postal Code'].copy()
    abbv_dict['Guam'] = "GU"
    labels = [abbv_dict[c] for c in min_wage_corr.columns]  # get abbv state names.
    # ******************
    print(abbv_dict_)
    print(abbv_dict)
    print (min_wage_corr.columns)
    print(labels)
    # 5.9 plot with new labels
    # looks nice in Jupyter but not in Python
    print("\n5.9:\n")
    fig = plt.figure(figsize=(12, 12))  # figure so we can add axis
    ax = fig.add_subplot(111)  # define axis, so we can modify
    ax.matshow(min_wage_corr, cmap=plt.cm.RdYlGn)  # display the matrix
    ax.set_xticks(np.arange(len(labels)))  # show them all!
    ax.set_yticks(np.arange(len(labels)))  # show them all!
    ax.set_xticklabels(labels)  # set to be the abbv (vs useless #)
    ax.set_yticklabels(labels)  # set to be the abbv (vs useless #)
    plt.show()


def six_combiningDatasets():
    """
    The join() method combines two DataFrames by index.
    The merge() method combines two DataFrames by whatever column you specify.
    """
    # 6.1 load data from four(), remove records with zero
    print("\n6.1:\n")
    # ******************
    df = pd.read_csv("minwage.csv")
    max_rows = 5  # None
    max_cols = None
    act_min_wage = pd.DataFrame()
    for name, group in df.groupby("State"):
        if act_min_wage.empty:
            with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols, ):
                print("\n6.1.1:\nname, group: ", name, "\n", group, "\n")
            # the group data for one state:
            # set index Year, filter for one column State.Minimum.Wage, rename the column to state name
            act_min_wage = group.set_index("Year")[["State.Minimum.Wage"]].rename(columns={"State.Minimum.Wage": name})
        else:
            act_min_wage = act_min_wage.join(
                group.set_index("Year")[["State.Minimum.Wage"]].rename(columns={"State.Minimum.Wage": name}))
    act_min_wage = act_min_wage.replace(0, np.NaN).dropna(axis=1)
    # ******************
    print(act_min_wage.head())
    # 6.2 load unemployment data
    print("\n6.2:\n")
    # ******************
    unemp_county = pd.read_csv("county_unemployment.csv")
    # ******************
    print(unemp_county.head())
    # 6.3 some ways to access data in df
    print("\n6.3:\n")
    print(act_min_wage["Colorado"].loc[2012])
    print(act_min_wage["Colorado"][2012])
    print(act_min_wage.loc[2012]["Colorado"])
    print(act_min_wage.loc[2012])
    print(act_min_wage["Colorado"])
    # 6.4 We can use this method to map just about any function with as many parameters
    # as we want to a column. This method will basically always work,
    # but won't necessarily be the most efficient. Often, we can use .map or .apply
    # instead to a column, or some other built-in methods, but the above is always an option.
    print("\n6.4:\n")
    # ******************
    global global_act_min_wage
    global_act_min_wage = act_min_wage
    with six_timeit_context('\n6.4.1: time to map\n'):
        if 0:
            unemp_county['min_wage'] = list(map(six_get_min_wage, unemp_county['Year'], unemp_county['State']))
            unemp_county.to_csv("unemployment_minwage_county.csv")
        else:
            unemp_county = pd.read_csv("unemployment_minwage_county.csv", index_col=0)
    # ******************
    print('\n6.4.2:\n', unemp_county.head())
    print('\n6.4.3:\n', unemp_county.tail())
    # 6.5 variances and correlations
    """ It looks like there's a slightly positive relationship (correlation) 
    between the unemployment rate and minimum wage, but also a pretty strong covariance, 
    signaling to us that these two things do tend to vary together. 
    It just looks like, while they definitely vary together, 
    the actual impact of one on the other isn't very substantial. 
    Plus, we'd have to ask next which comes first. 
    The increased unemployment, or the minimum wage increases."""
    print("\n6.5:\n")
    # ******************
    print(unemp_county[['Rate', 'min_wage']].corr())
    print(unemp_county[['Rate', 'min_wage']].cov())
    # ******************
    # variance on same State.Minimum.Wage numbers but mapped into different datasets
    print(df[['State.Minimum.Wage']].var())
    print(act_min_wage.var())
    print(act_min_wage.var().var())
    # 6.6 more data: add election data
    # Finally, I'd like to look at election data by county
    # and see if there's a relationship between voting, minimum wage, and unemployment.
    print("\n6.6:\n")
    # ******************
    pres16 = pd.read_csv("pres16results.csv")
    # ******************
    print("\n6.6:\n", pres16.head(15))
    # 6.6.1 top candidates
    top_candidates = pres16.head(10)['cand'].values
    print("\n6.6.1:\n", top_candidates)
    # 6.6.2 filter unemployment 2015 February
    # ******************
    county_2015 = unemp_county[(unemp_county['Year'] == 2015) & (unemp_county["Month"] == "February")].copy()
    # ******************
    print("\n6.6.2:\n", county_2015.head())
    # 6.6.3 read csv with state abbreviations; columns: 'State Name/District' and 'Postal Code'
    # ******************
    state_abbv = pd.read_csv("state_abbv.csv", index_col=0)
    # ******************
    print("\n6.6.3:\n", state_abbv.head())
    # 6.6.4 create dictionary with key 'State Name/District' and 'valuePostal Code'
    state_abbv_dict = state_abbv.to_dict()['Postal Code']
    print(state_abbv_dict)
    # 6.6.5 Map: take column State and map abbreviation into same column State; i.e overwrite
    # In the case of singe-parmeter functions, we can just use a .map.
    # Or...as you just saw here, if you want to map a key to a value using a dict,
    # you can do the same thing, and just say you want to map the dictionary.
    # ******************
    county_2015['State'] = county_2015['State'].copy().map(state_abbv_dict)
    # ******************
    print("\n6.6.5:\n", county_2015.tail())
    # 6.6.6 Since pres16 is longer, we'll map that to county_15, where there are matches.
    # Instead of a map, however, we'll combine with a join.
    # To do this, let's index both of these. They are indexed by state AND county.
    # So, we'll name these both the same, and then index as such.
    print("\n6.6.6.1:\n", len(county_2015))
    print("\n6.6.6.2:\n", len(pres16))
    # ******************
    pres16.rename(columns={"county": "County", "st": "State"}, inplace=True)
    # ******************
    print("\n6.6.6.3:\n", pres16.head())
    # 6.6.7 now index both df; df is just a variable to apply set_index to the dfs in brackets
    # ******************
    for df in [county_2015, pres16]:
        df.set_index(["County", "State"], inplace=True)
    # ******************
    # filter for records of candidate DJT cand = Donald Trump
    # only take index columns and column pct,
    # which shows the percentage of received share of votes, e.g 0.7 = 70%; and drop NaN records
    # ******************
    pres16 = pres16[pres16['cand'] == "Donald Trump"]
    pres16 = pres16[['pct']]
    pres16.dropna(inplace=True)
    # ******************
    print("\n6.6.7:\n", pres16.head(2))
    # 6.6.8 now merge both df on indexes; drop records with NaN; drop column Year
    # ******************
    all_together = county_2015.merge(pres16, on=["County", "State"])
    all_together.dropna(inplace=True)
    all_together.drop("Year", axis=1, inplace=True)
    # ******************
    print("\n6.6.8.1:\n", all_together.head())
    print("\n6.6.8.2:\n", county_2015.head(2))
    # 6.7 correlation and covariance
    print("\n6.7:\n")
    # ******************
    print(all_together.corr())
    print(all_together.cov())
    # ******************


def six_get_min_wage(year, state):
    try:
        return global_act_min_wage.loc[year][state]
    except:
        return np.NaN
    finally:
        pass


# Contextmanager to measure time
@contextmanager
def six_timeit_context(name):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print('[{}] finished in {} ms'.format(name, int(elapsed_time * 1_000)))


def seven_predictDiamondPrices():
    # load data into df
    # ******************
    df = pd.read_csv("diamonds.csv", index_col=0)
    # ******************
    print(df.tail())
    # mapping: transform text columns into numbers
    print(df['cut'].unique())
    print(df['color'].unique())
    print(df['clarity'].unique())
    # ******************
    cut_class_dict = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    color_dict = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
    clarity_dict = {"I3": 1, "I2": 2, "I1": 3, "SI2": 4, "SI1": 5, "VS2": 6,
                    "VS1": 7, "VVS2": 8, "VVS1": 9, "IF": 10, "FL": 11}
    df['cut'] = df['cut'].map(cut_class_dict)
    df['clarity'] = df['clarity'].map(clarity_dict)
    df['color'] = df['color'].map(color_dict)
    # ******************
    print(df.head())
    # prepare an SGD Regressor
    # ******************
    df = sklearn.utils.shuffle(df)  # always shuffle your data to avoid any biases that may emerge b/c of some order.
    X = df.drop("price", axis=1).values  # .values: to convert to a numpy array
    X = preprocessing.scale(X)  # always scale data; either here or before training
    y = df["price"].values
    test_size = 200
    X_train = X[:-test_size]  #:negative index: from zero to index before end
    y_train = y[:-test_size]
    X_test = X[-test_size:]  #:negative index: from index before end to end
    y_test = y[-test_size:]
    # train SGD Regressor
    # Always scale the input. The most convenient way is to use a pipeline.
    # clf = SGDRegressor(max_iter=1000)
    clf = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3))
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))  # R-Squared
    # price prediction vs real price
    for X, y in list(zip(X_test, y_test))[:10]:
        print(f"model predicts {clf.predict([X])[0]}, real value: {y}")
    # ******************
    # prepare an svm
    # clf = svm.SVR()
    clf = make_pipeline(StandardScaler(), svm.SVR(C=1.0, epsilon=0.2, max_iter=1000))
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    # price prediction vs real price
    for X, y in list(zip(X_test, y_test))[:10]:
        print(f"model predicts {clf.predict([X])[0]}, real value: {y}")


if __name__ == '__main__':
    if 0:
        one_filterForRegion_setDateAsIndex()
        two_convertToDatetime_sortDate_rollingMean_toMarkdown()
        three_filterForType_rollingMeanPerRegion()
        four_groupBy_setYearAsIndex_removeZeros_statistics()
        five_visualizeCorrelations_getLabelsFromWebsiteTables()
        six_combiningDatasets()
    else:
        seven_predictDiamondPrices()
