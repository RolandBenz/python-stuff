import pandas as pd
import numpy as np
from tabulate import tabulate
import string
import random


# Tutorial from:
# https://www.statology.org/
# https://stackoverflow.com/questions/26886653/pandas-create-new-column-based-on-values-from-other-columns-apply-a-function-o


def one_df_join():
    # 1. df.join()
    """used only on indexes of df and df"""
    # create two DataFrames
    df1 = pd.DataFrame({'name': ['A', 'B', 'C'], 'points': [8, 12, 19]}).set_index('name')
    df2 = pd.DataFrame({'name': ['A', 'B', 'C'], 'steals': [4, 5, 2]}).set_index('name')
    # view two DataFrames
    print("\n1.1.1\n", df1)
    print("1.1.2\n", df2)
    # use join() function to join together two DataFrames
    print("\n1.2.1\n", df1.join(df2))
    # view DataFrame; no longer joined because assignment missing
    print("\n1.2.2\n", df1)
    # use join() function to join together two DataFrames
    # ******************
    df_j = df1.join(df2)
    # ******************
    # view joined DataFrame
    print("\n1.3\n", df_j)
    # use join() function to join together two DataFrames
    # ******************
    df1 = df1.join(df2)
    # ******************
    # view DataFrame
    print("\n1.4\n", df1)


def two_df_merge():
    # 2. df.merge()
    """used on any column of df and df"""
    # create two DataFrames
    df1 = pd.DataFrame({'name': ['A', 'B', 'C'], 'points': [8, 12, 19]}).set_index('name')
    df2 = pd.DataFrame({'name': ['A', 'B', 'C'], 'steals': [4, 5, 2]}).set_index('name')
    # view two DataFrames
    print("\n2.1.1\n", df1);
    print("2.1.2", df2)
    # use merge() function to join together two DataFrames
    # ******************
    df1_m = df1.merge(df2, on='name', how='left')
    df1_m_ = pd.merge(df1, df2, on='name', how='left')
    # ******************
    # view DataFrame
    print("\n2.2\n", df1_m)
    print("\n2.2_\n", df1_m_)
    #
    # another example
    df = pd.DataFrame.from_dict({
        'Name': ['Nik', 'Kate', 'James', 'Nik', 'Kate', 'James', 'Nik', 'Kate', 'James'],
        'Month': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'Gender': ['M', 'F', 'M', 'M', 'F', 'M', 'M', 'F', 'M']
    })
    print("\n2.3\n", df)
    months = pd.DataFrame.from_dict({
        'Number': [1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'Months': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December']
    })
    print("\n2.4\n", months)
    #
    # join explained
    # ******************
    df_m1_ = df.merge(months, left_on='Month', right_on='Number', how='left')
    df_m1 = pd.merge(
        left=df,  # df left side
        right=months,  # df right side
        left_on='Month',  # which pandas columns to use to merge the dataframes.
        right_on='Number',  # which pandas columns to use to merge the dataframes.
        # Use on= If the columns are the same across the dataframes.
        how='left'  # how to merge the tables (either a left, right, inner, or outer join)
    )
    # ******************
    print("\n2.5.1_: left\n", df_m1_)
    print("\n2.5.1: left\n", df_m1)
    #
    # ******************
    df_m2 = pd.merge(
        left=df,  # df left side
        right=months,  # df right side
        left_on='Month',  # which pandas columns to use to merge the dataframes.
        right_on='Number',  # which pandas columns to use to merge the dataframes.
        # Use on= If the columns are the same across the dataframes.
        how='right'  # how to merge the tables (either a left, right, inner, or outer join)
    )
    # ******************
    print("\n2.5.2: right\n", df_m2)
    #
    # ******************
    df_m3 = pd.merge(
        left=df,  # df left side
        right=months,  # df right side
        left_on='Month',  # which pandas columns to use to merge the dataframes.
        right_on='Number',  # which pandas columns to use to merge the dataframes.
        # Use on= If the columns are the same across the dataframes.
        how='inner'  # how to merge the tables (either a left, right, inner, or outer join)
    )
    # ******************
    print("\n2.5.3: inner\n", df_m3)
    #
    # ******************
    df_m4 = pd.merge(
        left=df,  # df left side
        right=months,  # df right side
        left_on='Month',  # which pandas columns to use to merge the dataframes.
        right_on='Number',  # which pandas columns to use to merge the dataframes.
        # Use on= If the columns are the same across the dataframes.
        how='outer'  # how to merge the tables (either a left, right, inner, or outer join)
    )
    # ******************
    print("\n2.5.4: outer\n", df_m4)
    #
    # ******************
    df_m5 = pd.merge(
        left=months,  # df left side
        right=df,  # df right side
        left_on='Number',  # which pandas columns to use to merge the dataframes.
        right_on='Month',  # which pandas columns to use to merge the dataframes.
        # Use on= If the columns are the same across the dataframes.
        how='left'  # how to merge the tables (either a left, right, inner, or outer join)
    )
    # ******************
    print("\n2.6.1: left\n", df_m5)
    #
    # ******************
    df_m6 = pd.merge(
        left=months,  # df left side
        right=df,  # df right side
        left_on='Number',  # which pandas columns to use to merge the dataframes.
        right_on='Month',  # which pandas columns to use to merge the dataframes.
        # Use on= If the columns are the same across the dataframes.
        how='right'  # how to merge the tables (either a left, right, inner, or outer join)
    )
    # ******************
    print("\n2.6.2: right\n", df_m6)
    #
    # ******************
    df_m7 = pd.merge(
        left=months,  # df left side
        right=df,  # df right side
        left_on='Number',  # which pandas columns to use to merge the dataframes.
        right_on='Month',  # which pandas columns to use to merge the dataframes.
        # Use on= If the columns are the same across the dataframes.
        how='inner'  # how to merge the tables (either a left, right, inner, or outer join)
    )
    # ******************
    print("\n2.6.3: inner\n", df_m7)
    #
    # ******************
    df_m8 = pd.merge(
        left=months,  # df left side
        right=df,  # df right side
        left_on='Number',  # which pandas columns to use to merge the dataframes.
        right_on='Month',  # which pandas columns to use to merge the dataframes.
        # Use on= If the columns are the same across the dataframes.
        how='outer'  # how to merge the tables (either a left, right, inner, or outer join)
    )
    # ******************
    print("\n2.6.4: outer\n", df_m8)


def three_df_map():
    # 3. df.map()
    """used on any column of df and dict"""
    df = pd.DataFrame.from_dict({
        'Name': ['Nik', 'Kate', 'James', 'Nik', 'Kate', 'James', 'Nik', 'Kate', 'James'],
        'Month': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'Gender': ['M', 'F', 'M', 'M', 'F', 'M', 'M', 'F', 'M']
    })
    print("\n3.1\n", df)
    gender_map = {
        'M': 'Male',
        'F': 'Female'
    }
    # map into new column Gender(Expanded) on column Gender; with dict gender_map
    # ******************
    df['Gender(Expanded)'] = df['Gender'].map(gender_map)
    # ******************
    print("\n3.2\n", df)


def four_df_append():
    # 4. df.append() - add record(s) at the end of df
    """used to add records of df to df"""
    # create DataFrame
    df = pd.DataFrame({'points': [10, 12, 12, 14, 13, 18],
                       'rebounds': [7, 7, 8, 13, 7, 4],
                       'assists': [11, 8, 10, 6, 6, 5]})
    # view DataFrame
    print("\n4.1\n", df)
    # add new row to end of DataFrame
    # ******************
    df.loc[len(df.index)] = [20, 7, 5]
    # ******************
    # view updated DataFrame
    print("\n4.2\n", df)
    # define second DataFrame
    df2 = pd.DataFrame({'points': [21, 25, 26],
                        'rebounds': [7, 7, 13],
                        'assists': [11, 3, 3]})
    # add new row to end of DataFrame
    # ******************
    df = df.append(df2, ignore_index=True)  # depreciated method
    df_c1 = pd.concat([df, df2], ignore_index=True)  # same as df.append
    df_c2 = pd.concat([df, df2], keys=['key1', 'key2'])  # keep individual indexes, label them
    df_c3 = pd.concat([df, df2], axis=1)  # add columns to same index
    # ******************
    # view updated DataFrame
    print("\n4.3.1\n", df)
    print("\n4.3.2\n", df_c1)
    print("\n4.3.3\n", df_c2)
    print("\n4.3.4\n", df_c3)


def five_pd_DataFrame():
    # 5. convert any data structure to df
    """used to convert Python list or dict, etc. into df"""
    # define list
    x = [4, 5, 8, 'Mavericks']
    # convert list to DataFrame
    # ******************
    df = pd.DataFrame(x).T
    # ******************
    # specify column names of DataFrame
    df.columns = ['Points', 'Assists', 'Rebounds', 'Team']
    # display DataFrame
    print("\n5.1\n", df)
    #
    # define list of lists
    big_list = [[6, 7, 12, 'Mavericks'],
                [4, 2, 1, 'Lakers'],
                [12, 4, 8, 'Spurs']]
    # convert list of lists into DataFrame
    # ******************
    df = pd.DataFrame(columns=['Points', 'Assists', 'Rebounds', 'Team'], data=big_list)
    # ******************
    # display DataFrame
    print("\n5.2.1\n", df)
    print("\n5.2.2\n", df.shape)
    #
    dict_ = {'points': [21, 25, 26],
             'rebounds': [7, 7, 13],
             'assists': [11, 3, 3]}
    # ******************
    df = pd.DataFrame(data=dict_)
    # ******************
    print("\n5.3\n", df)


def six_df_iloc():
    # 6. get range, e.g first column of a df
    """used to get range, e.g column of df into pd series or pd df"""
    # create DataFrame
    df = pd.DataFrame({'points': [25, 12, 15, 14, 19, 23, 25, 29],
                       'assists': [5, 7, 7, 9, 12, 9, 9, 4],
                       'rebounds': [11, 8, 10, 6, 6, 5, 9, 12]})
    # view DataFrame
    print("\n6.1\n", df)
    # get first column (and return a Series)
    # ******************
    first_col = df.iloc[:, 0]
    # ******************
    # view first column
    print("\n6.2\n", first_col)
    # check type of first_col: <class 'pandas.core.series.Series'>
    print("\n6.3\n", type(first_col))
    # get first column (and return a DataFrame)
    # ******************
    first_col = df.iloc[:, :1]
    # ******************
    # view first column
    print("\n6.4\n", first_col)
    # check type of first_col: <class 'pandas.core.frame.DataFrame'>
    print("\n6.5\n", type(first_col))
    # ******************
    range_ = df.iloc[3:6, 1:3]  # index starts at 0; from x: to :y, but without y
    # ******************
    # view first column
    print("\n6.6\n", range_)
    # check type of first_col: <class 'pandas.core.frame.DataFrame'>
    print("\n6.7\n", type(range_))
    # ******************
    range_1 = df.iloc[[0, 2, 4, 5], [0, 2]]
    # ******************
    # view first column
    print("\n6.8\n", range_1)
    # check type of first_col: <class 'pandas.core.frame.DataFrame'>
    print("\n6.9\n", type(range_1))
    # ******************
    range_2 = df.iloc[[lambda x: x.index % 2 == 0], [0, 1]]
    # ******************
    # view first column
    print("\n6.10\n", range_2)
    # check type of first_col: <class 'pandas.core.frame.DataFrame'>
    print("\n6.11\n", type(range_2))


def seven_df_head():
    # 7. get first column of a df
    """used to get row of df into pd series or pd df"""
    # create DataFrame
    df = pd.DataFrame({'points': [25, 12, 15, 14, 19, 23, 25, 29],
                       'assists': [5, 7, 7, 9, 12, 9, 9, 4],
                       'rebounds': [11, 8, 10, 6, 6, 5, 9, 12]})
    # view DataFrame
    print("\n7.1\n", df)
    # get first row of DataFrame
    # ******************
    print("\n7.2\n", df.head(1))
    # get first row of values for points and rebounds columns
    print("\n7.3\n", df[['points', 'rebounds']].head(1))
    # get first row where points < 20 and assists > 10
    print("\n7.4\n", df[(df.points < 20) & (df.assists > 10)].head(1))
    # ******************


def eight_np_to_pd_DataFrame():
    # 8. convert numpy to panda df
    """used to convert np to df"""
    # create NumPy array
    data = np.array([[1, 7, 6, 5, 6], [4, 4, 4, 3, 1]])
    # print class of NumPy array
    type(data)
    # convert NumPy array to pandas DataFrame
    # ******************
    df = pd.DataFrame(data=data)
    # ******************
    # print DataFrame
    print("\n8.1\n", df.to_markdown())
    # print class of DataFrame
    print("\n8.2\n", type(df))
    """We can also specify row names and column names for the DataFrame 
    by using the index and columns arguments, respectively."""
    # convert array to DataFrame and specify rows & columns
    # ******************
    df = pd.DataFrame(data=data, index=["r1", "r2"], columns=["A", "B", "C", "D", "E"])
    # ******************
    # print the DataFrame (with own print function)
    df_nice("8.3", df)


def nine_df_insert():
    # 9. insert column into a df
    """used to insert columns pd df"""
    # create DataFrame
    df = pd.DataFrame({'points': [25, 12, 15, 14, 19],
                       'assists': [5, 7, 7, 9, 12],
                       'rebounds': [11, 8, 10, 6, 6]})
    # view DataFrame
    print("\n9.1\n", df)
    # insert new column 'player' as first column
    player_vals = pd.Series(['A', 'B', 'C', 'D', 'E'])
    # ******************
    df.insert(loc=0, column='player', value=player_vals)
    # ******************
    # view DataFrame
    print("\n9.2\n", df)
    # insert new column 'partner' as third column
    player_vals = pd.Series(['Aa', 'Ba', 'Ca', 'Da', 'Ea'])
    # ******************
    df.insert(loc=1, column='partner', value=player_vals)
    # ******************
    # view DataFrame
    print("\n9.3\n", df)
    # insert new column 'total' as last column
    # ******************
    player_vals = pd.Series(df.apply(lambda row: row.points + row.assists + row.rebounds, axis=1))
    df.insert(loc=len(df.columns), column='total', value=player_vals)
    # ******************
    # view DataFrame
    print("\n9.4.1\n", player_vals)
    print("\n9.4.2\n", df)
    # insert new column 'remarks' as last column
    player_vals = pd.Series(['points wow', 'hmm', 'rebounds wow', 'partner wow', 'assists wow'])
    # ******************
    df.insert(loc=len(df.columns), column='remarks', value=player_vals)
    # ******************
    # view DataFrame
    print("\n9.5\n", df)


def ten_df_assign():
    # 10. insert column at the end of a df
    """used to insert column at the end of pd df"""
    # create DataFrame
    df = pd.DataFrame({'points': [25, 12, 15, 14, 19, 23],
                       'assists': [5, 7, 7, 9, 12, 9],
                       'rebounds': [11, 8, 10, 6, 6, 5]})
    # view DataFrame
    print("\n10.1\n", df)
    # add 'steals' column to end of DataFrame
    # ******************
    df = df.assign(steals=[2, 2, 4, 7, 4, 1])
    # ******************
    # view DataFrame
    print("\n10.2\n", df)
    # add 'steals' and 'blocks' columns to end of DataFrame
    # ******************
    df = df.assign(steals_=[2, 2, 4, 7, 4, 1],
                   blocks=[0, 1, 1, 3, 2, 5])
    # ******************
    # view DataFrame
    print("\n10.3\n", df)
    # add 'half_pts' to end of DataFrame
    # ******************
    df = df.assign(extra=lambda x: x.points / 2 + (x.steals + x.rebounds) * 2)
    # ******************
    # view DataFrame
    print("\n10.4\n", df)


def eleven_df_renameColumns():
    # 11. rename columns of a df
    """used to rename columns pd df"""
    # define DataFrame
    df = pd.DataFrame({'team': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
                       'points': [25, 12, 15, 14, 19, 23, 25, 29],
                       'assists': [5, 7, 7, 9, 12, 9, 9, 4],
                       'rebounds': [11, 8, 10, 6, 6, 5, 9, 12]})
    # list column names
    print("\n11.1\n", list(df))
    # rename specific column names
    # ******************
    df.rename(columns={'team': 'team_name', 'points': 'points_scored'}, inplace=True)
    # ******************
    # view updated list of column names
    print("\n11.2\n", list(df))
    # view DataFrame
    print("\n11.3\n", df)
    # rename all column names
    # ******************
    df.columns = ['_team', '_points', '_assists', '_rebounds']
    # ******************
    # view DataFrame
    print("\n11.4\n", df)


def twelve_df_sortColumns():
    # 12. sort columns of a df
    """used to sort columns pd df"""
    # create DataFrame
    df = pd.DataFrame({'points': [25, 12, 15, 14, 19, 23, 25, 29],
                       'assists': [5, 7, 7, 9, 12, 9, 9, 4],
                       'rebounds': [11, 8, 10, 6, 6, 5, 9, 12],
                       'steals': [2, 3, 3, 2, 5, 3, 2, 1]})
    # list column names
    print("\n12.1\n", list(df))
    # sort columns by names
    # ******************
    df = df[['steals', 'assists', 'rebounds', 'points']]
    # ******************
    # view DataFrame
    print("\n12.2\n", df)
    # define list of column names
    name_order = ['steals', 'assists', 'points', 'rebounds']
    # sort columns by list
    # ******************
    df = df[name_order]
    # ******************
    # view DataFrame
    print("\n12.3\n", df)
    # sort columns alphabetically
    # ******************
    df = df[sorted(df.columns)]
    # ******************
    # view DataFrame
    print("\n12.4\n", df)


def thirteen_df_unique():
    # 13. unique values of a df
    """used to get unique values in columns of df"""
    # create DataFrame
    df = pd.DataFrame({'team': ['A', 'A', 'A', 'B', 'B', 'C'],
                       'conference': ['East', 'East', 'East', 'West', 'West', 'East'],
                       'points': [11, 8, 10, 6, 6, 5]})
    # view DataFrame
    print("\n13.1\n", df)
    # unique values in one column
    # count occurrence of values in a column
    # unique values in all columns
    # ******************
    print("\n13.2\n", df.team.unique())
    print("\n13.3\n", df.team.value_counts())
    for col in df:
        print("\n13.4\n", df[col].unique())
    # ******************
    # sort unique values
    points = df.points.unique()
    points.sort()
    # display sorted values
    print("\n13.5\n", points)


def fourteen_df_prettyPrint():
    df = pd.DataFrame.from_dict({
        'Name': ['Nik', 'Kate', 'James', 'Nik', 'Kate', 'James', 'Nik', 'Kate', 'James'],
        'Month': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'Gender': ['M', 'F', 'M', 'M', 'F', 'M', 'M', 'F', 'M'],
        'team': ['A', 'A', 'A', 'B', 'B', 'C', 'B', 'B', 'C'],
        'conference': ['East', 'East', 'East', 'West', 'West', 'East', 'West', 'West', 'East'],
        'points': [11, 8, 10, 6, 6, 5, 6, 6, 5],
        'assists': [5, 7, 7, 9, 12, 9, 9, 4, 4],
        'rebounds': [11, 8, 10, 6, 6, 5, 9, 12, 12],
        'steals': [2, 3, 3, 2, 5, 3, 2, 1, 1]
    })
    if 0:
        print(df.to_markdown())
        print(df.to_html('temp.html'))
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    else:
        df_nice("14.", df)


def df_nice(tx, df):
    print('\n', tx, '\n')
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=True))


def fifteen_mergeJoinConcat_onIndex():
    # create first DataFrame
    df1 = pd.DataFrame({'rating': [90, 85, 82, 88, 94, 90, 76, 75],
                        'points': [25, 20, 14, 16, 27, 20, 12, 15]},
                       index=list('abcdefgh'))
    df_nice("15.1", df1)
    # create second DataFrame
    df2 = pd.DataFrame({'assists': [5, 7, 7, 8, 5, 7],
                        'rebounds': [11, 8, 10, 6, 6, 9]},
                       index=list('acdgmn'))
    df_nice("15.2", df2)
    # join df
    # ******************
    df_j = df1.join(df2)  # default: left join
    df_m = pd.merge(df1, df2, left_index=True, right_index=True)  # default: inner join
    df_c = pd.concat([df1, df2], axis=1)  # default: outer join
    # ******************
    df_nice("15.3", df_j)
    df_nice("15.4", df_m)
    df_nice("15.5", df_c)


def sixteen_mergeOnMultipleColumns():
    # create and view first DataFrame
    df1 = pd.DataFrame({'a1': [0, 0, 1, 1, 2],
                        'b': [0, 0, 1, 1, 1],
                        'c': [11, 8, 10, 6, 6]})
    df_nice("16.1", df1)
    # create and view second DataFrame
    df2 = pd.DataFrame({'a2': [0, 1, 1, 1, 3],
                        'b': [0, 0, 0, 1, 1],
                        'd': [22, 24, 25, 33, 37]})
    df_nice("16.2", df2)
    # default with .merge(): inner join; here with how= changed to left join
    # ******************
    df_m = pd.merge(df1, df2, how='left', left_on=['a1', 'b'], right_on=['a2', 'b'])
    # ******************
    # create DataFrames
    df1 = pd.DataFrame({'a': [0, 0, 1, 1, 2],
                        'b': [0, 0, 1, 1, 1],
                        'c': [11, 8, 10, 6, 6]})
    df2 = pd.DataFrame({'a': [0, 1, 1, 1, 3],
                        'b': [0, 0, 1, 1, 1],
                        'd': [22, 24, 25, 33, 37]})
    # ******************
    df_m1 = pd.merge(df1, df2, how='left', on=['a', 'b'])
    # ******************
    df_nice("16.3", df_m)
    df_nice("16.4", df_m1)


def seventeen_systematicSampling():
    # make this example reproducible
    np.random.seed(0)
    # create DataFrame
    # ******************
    df = pd.DataFrame(
        {'last_name': [''.join(random.choice(string.ascii_uppercase) for _ in range(6)) for _ in range(500)],
         'GPA': np.random.normal(loc=85, scale=3, size=500)})
    # ******************
    # view last six rows of DataFrame
    print("\n17.1\n", df.tail())
    # obtain systematic sample by selecting every 5th row
    # ******************
    sys_sample_df = df.iloc[::5]
    # ******************
    # view last six rows of DataFrame
    print("\n17.2\n", sys_sample_df.tail())
    # view dimensions of data frame
    print("\n17.3\n", sys_sample_df.shape)


def eighteen_dropRowsThatSatisfyCondition():
    # create DataFrame
    df = pd.DataFrame({'team': ['A', 'A', 'A', 'B', 'B', 'C'],
                       'conference': ['South', 'East', 'North', 'West', 'West', 'East'],
                       'points': [11, 8, 10, 6, 6, 5]})
    # view DataFrame
    df_nice("18.1", df)
    # drop rows
    # ******************
    df_c1 = df[df["team"].str.contains("A") == False]
    df_c2 = df[df["team"].str.contains("A|B") == False]
    # ******************
    # identify partial string to look for
    # drop (~ means false or not) rows that contain the partial string "Wes|ast" in the conference column
    discard = ["Wes", "ast"]
    filter_ = '|'.join(discard)
    # ******************
    df_c3 = df[~df.conference.str.contains(filter_)]
    # ******************
    df_nice("18.2", df_c1)
    df_nice("18.3", df_c2)
    print("18.4.1\n", discard, '\n', filter_)
    df_nice("18.4.2", df_c3)


def nineteen_combineColumns():
    # create dataFrame
    df = pd.DataFrame({'team': ['Mavs', 'Lakers', 'Spurs', 'Cavs'],
                       'first': ['Dirk', 'Kobe', 'Tim', 'Lebron'],
                       'last': ['Nowitzki', 'Bryant', 'Duncan', 'James'],
                       'points': [26, 31, 22, 29]})
    df_nice("19.1", df)
    # combine first and last name column into new column, with space in between
    # ******************
    df.insert(loc=3, column='full_name', value=df['first'] + ' ' + df['last'])
    df['full_name_'] = df['first'] + '_' + df['last']
    df['name_points'] = df['last'] + df['points'].astype(str)
    df['team_and_name_'] = df['team'] + '_' + df['first'] + '_' + df['last']
    df['team_and_name'] = df[['team', 'first', 'last']].agg(' '.join, axis=1)
    # ******************
    # view resulting dataFrame
    df_nice("19.2", df)


def twenty_filterRowsThatSatisfyCondition():
    # create DataFrame
    df = pd.DataFrame({'team': ['A', 'A', 'A', 'B', 'B', 'C'],
                       'conference': ['South', 'East', 'North', 'West', 'West', 'East'],
                       'points': [11, 8, 10, 6, 6, 5]})
    # view DataFrame
    df_nice("20.1", df)
    # ******************
    df_c1 = df[df["team"].str.contains("A")]
    df_c2 = df[df["team"].str.contains("A|B")]
    df_c3 = df[df.conference.str.contains('|'.join(["Wes", "ast"]))]
    # ******************
    df_nice("20.2", df_c1)
    df_nice("20.3", df_c2)
    df_nice("20.4", df_c3)


def twentyone_groupBy_aggregate():
    # create DataFrame
    df = pd.DataFrame({'store': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
                       'quarter': [1, 1, 2, 2, 1, 1, 2, 2],
                       'employee': ['Andy', 'Bob', 'Chad', 'Diane',
                                    'Elana', 'Frank', 'George', 'Hank']})
    # view DataFrame
    df_nice("21.1", df)
    # group by store and quarter, then concatenate employee strings
    # ******************
    df_c1 = df.groupby(['store', 'quarter'], as_index=False)
    df_c2 = df.groupby(['store', 'quarter'], as_index=False).agg({'employee': ' & '.join})
    # ******************
    df_nice("21.2", df_c1)
    df_nice("21.3", df_c2)


def twentytwo_readHtmlTables():
    # read all HTML tables from specific URL
    # ******************
    tabs = pd.read_html('https://en.wikipedia.org/wiki/National_Basketball_Association')
    # ******************
    # display total number of tables read
    print("\n22.1\n", len(tabs))
    # read HTML tables from specific URL with the word "Division" in them
    # ******************
    tabs = pd.read_html('https://en.wikipedia.org/wiki/National_Basketball_Association',
                        match='Division')
    df = tabs[0]
    # ******************
    # display total number of tables read
    print("\n22.2\n", len(tabs))
    # list all column names of table
    print("\n22.3\n", list(df))
    # filter DataFrame to only contain first two columns
    # ******************
    df_final = df.iloc[:, 0:2]
    # ******************
    # rename columns
    df_final.columns = ['Division', 'Team']
    # view first few rows of final DataFrame
    df_nice("22.4", df_final)


def twentythree_crossJoin():
    # create first DataFrame
    df1 = pd.DataFrame({'team': ['A', 'B', 'C', 'D'],
                        'points': [18, 22, 19, 14]})
    df_nice("23.1", df1)
    # create second  DataFrame
    df2 = pd.DataFrame({'team': ['A', 'B', 'F'],
                        'assists': [4, 9, 8]})
    df_nice("23.2", df2)
    # create common key
    # perform cross join
    # ******************
    df1['key'] = 0
    df2['key'] = 0
    df3 = df1.merge(df2, on='key', how='outer')
    # ******************
    df_nice("23.3", df3)
    # drop key columm
    del df3['key']
    # view results
    df_nice("23.4", df3)


def twentyfour_pivotTable():
    """with aggregation of values, along index and values"""
    df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                       "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                       "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                       "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                       "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
    df_nice("24.1", df)
    #
    # index: row key; columns: column titles;
    # values: aggregated table data; aggfunc: how values are calculated, e.g. summed up or something else
    # ******************
    pivottable1 = pd.pivot_table(df, index=['A', 'B'], columns=['C'],
                                 values='D', aggfunc=np.sum)
    pivottable2 = pd.pivot_table(df, index=['A', 'B'], columns=['C'],
                                 values='D', aggfunc=np.sum, fill_value=0)
    pivottable3 = pd.pivot_table(df, index=['A', 'C'],
                                 values=['D', 'E'], aggfunc={'D': np.mean, 'E': np.mean})
    pivottable4 = pd.pivot_table(df, index=['A', 'C'],
                                 values=['D', 'E'], aggfunc={'D': np.mean, 'E': [min, max, np.mean]})
    # ******************
    df_nice("24.2", pivottable1)
    df_nice("24.3", pivottable2)
    df_nice("24.4", pivottable3)
    df_nice("24.5", pivottable4)


def twentyfive_pivot():
    """no aggregation of values"""
    df1 = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                        'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                        'baz': [1, 2, 3, 4, 5, 6],
                        'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
    df2 = pd.DataFrame({"lev1": [1, 1, 1, 2, 2, 2],
                        "lev2": [1, 1, 2, 1, 1, 2],
                        "lev3": [1, 2, 1, 2, 1, 2],
                        "lev4": [1, 2, 3, 4, 5, 6],
                        "values": [0, 1, 2, 3, 4, 5]})
    df_nice("25.1", df1)
    df_nice("25.2", df2)
    # ******************
    pivot1 = df1.pivot(index='foo', columns='bar', values='baz')
    pivot2 = df1.pivot(index='foo', columns='bar')['baz']
    pivot3 = df1.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
    pivot4 = df2.pivot(index="lev1", columns=["lev2", "lev3"], values="values")
    pivot5 = df2.pivot(index=["lev1", "lev2"], columns=["lev3"], values="values")
    # ******************
    df_nice("25.3", pivot1)
    df_nice("25.4", pivot2)
    df_nice("25.5", pivot3)
    df_nice("25.6", pivot4)
    df_nice("25.7", pivot5)


def twentysix_df_compareColumns():
    # create first DataFrame
    df1 = pd.DataFrame({'team': ['Mavs', 'Rockets', 'Spurs', 'Heat', 'Nets', 'Yanks'],
                        'points': [22, 30, 15, 17, 14, 17]})
    # create second DataFrame
    df2 = pd.DataFrame({'team': ['Mavs', 'Thunder', 'Spurs', 'Nets', 'Cavs'],
                        'points': [25, 40, 31, 32, 22]})
    # view DataFrame
    df_nice("26.1", df1)
    df_nice("26.2", df2)
    # count matching values in team columns
    # display matching values between team columns (use: inner join)
    # ******************
    count_ = df1['team'].isin(df2['team']).value_counts()
    df_show_ = pd.merge(df1, df2, on=['team'], how='inner')
    # ******************
    print("\n26.3\n", count_)
    df_nice("26.4", df_show_)


def twentyseven_fillColumnWithValues():
    # make a simple dataframe
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df_nice("27.1", df)
    # 1a. create an unattached column with an index
    # 1b. do same but attach it to the dataframe
    # ******************
    print("\n27.2\n", df.apply(lambda row: row.a + row.b, axis=1))
    df['c'] = df.apply(lambda row: row.a + row.b, axis=1)
    # ******************
    df_nice("27.3", df)
    #
    # 1c. do the same in separate steps
    # ******************
    fn = lambda row: row.a + row.b  # define a function for the new column
    col = df.apply(fn, axis=1)  # get column data with an index
    df = df.assign(c=col.values)  # assign values to column 'c'
    # ******************
    df_nice("27.4", df)


def twentyeight_fillColumnWithValues_OnCondition_v1():
    big_list = [['MOST', 'JEFF', 'E', 0, 0, 0, 0, 0, 1, 'White'],
                ['CRUISE', 'TOM', 'E', 0, 0, 0, 1, 0, 0, 'White'],
                ['DEPP', 'JOHNNY', '', 0, 0, 0, 0, 0, 1, 'Unknown'],
                ['DICAP', 'LEO', '', 0, 0, 0, 0, 0, 1, 'Unknown'],
                ['BRANDO', 'MARLON', 'E', 0, 0, 0, 0, 0, 0, 'White'],
                ['HANKS', 'TOM', '', 0, 0, 0, 0, 0, 1, 'Unknown'],
                ['DENIRO', 'ROBERT', 'E', 0, 1, 0, 0, 0, 1, 'White'],
                ['PACINO', 'AL', 'E', 0, 0, 0, 0, 0, 1, 'White'],
                ['WILLIAMS', 'ROBIN', 'E', 0, 0, 1, 0, 0, 0, 'White'],
                ['EASTWOOD', 'CLINT', 'E', 0, 0, 0, 0, 0, 1, 'White']]

    # convert list of lists into DataFrame
    df = pd.DataFrame(columns=['lname', 'fname', 'rno_cd', 'eri_afr_amer',
                               'eri_asian', 'eri_hawaiian', 'eri_hispanic',
                               'eri_nat_amer', 'eri_white', 'rno_defined'], data=big_list)
    df_nice("28.1", df)
    print("\n28.2\n", df.apply(lambda row: twentyeight_label_race(row), axis=1))
    # ******************
    df['race_label'] = df.apply(lambda row: twentyeight_label_race(row), axis=1)
    df['race_label_'] = df.apply(twentyeight_label_race, axis=1)
    # ******************
    df_nice("28.3", df)


def twentyeight_label_race(row):
    if row['eri_afr_amer'] + row['eri_asian'] + row['eri_hawaiian'] + \
            row['eri_nat_amer'] + row['eri_white'] > 1:
        return 'Two Or More'
    if row['eri_hispanic'] == 1:
        return 'Hispanic'
    if row['eri_nat_amer'] == 1:
        return 'A/I AK Native'
    if row['eri_asian'] == 1:
        return 'Asian'
    if row['eri_afr_amer'] == 1:
        return 'Black/AA'
    if row['eri_hawaiian'] == 1:
        return 'Haw/Pac Isl.'
    if row['eri_white'] == 1:
        return 'White'
    return 'Other'


def twentynine_fillColumnWithValues_OnCondition_v2():
    big_list = [['MOST', 'JEFF', 'E', 0, 0, 0, 0, 0, 1, 'White'],
                ['CRUISE', 'TOM', 'E', 0, 0, 0, 1, 0, 0, 'White'],
                ['DEPP', 'JOHNNY', '', 0, 0, 0, 0, 0, 1, 'Unknown'],
                ['DICAP', 'LEO', '', 0, 0, 0, 0, 0, 1, 'Unknown'],
                ['BRANDO', 'MARLON', 'E', 0, 0, 0, 0, 0, 0, 'White'],
                ['HANKS', 'TOM', '', 0, 0, 0, 0, 0, 1, 'Unknown'],
                ['DENIRO', 'ROBERT', 'E', 0, 1, 0, 0, 0, 1, 'White'],
                ['PACINO', 'AL', 'E', 0, 0, 0, 0, 0, 1, 'White'],
                ['WILLIAMS', 'ROBIN', 'E', 0, 0, 1, 0, 0, 0, 'White'],
                ['EASTWOOD', 'CLINT', 'E', 0, 0, 0, 0, 0, 1, 'White']]

    # convert list of lists into DataFrame
    df = pd.DataFrame(columns=['lname', 'fname', 'rno_cd', 'eri_afr_amer',
                               'eri_asian', 'eri_hawaiian', 'eri_hispanic',
                               'eri_nat_amer', 'eri_white', 'rno_defined'], data=big_list)
    df_nice("29.1", df)
    #
    # use numpy, it is faster
    conditions = [
        df[['eri_afr_amer', 'eri_asian', 'eri_hawaiian', 'eri_nat_amer', 'eri_white']].sum(1).gt(1),
        df['eri_afr_amer'] == 1,
        df['eri_asian'] == 1,
        df['eri_hawaiian'] == 1,
        df['eri_hispanic'] == 1,
        df['eri_nat_amer'] == 1,
        df['eri_white'] == 1,
    ]
    outputs = [
        'Two Or More', 'Black/AA', 'Asian', 'Haw/Pac Isl.', 'Hispanic', 'A/I AK Native', 'White'
    ]
    # ******************
    res = np.select(conditions, outputs, 'Other')
    sres = pd.Series(res)
    df.insert(loc=len(df.columns), column='total', value=sres)
    # ******************
    print("\n29.2\n", res)
    print("\n29.3\n", sres)
    df_nice("29.4", df)


def thirty_fillColumnWithValues_OnCondition_v3():
    big_list = [['MOST', 'JEFF', 'E', 0, 0, 0, 0, 0, 1, 'White'],
                ['CRUISE', 'TOM', 'E', 0, 0, 0, 1, 0, 0, 'White'],
                ['DEPP', 'JOHNNY', '', 0, 0, 0, 0, 0, 1, 'Unknown'],
                ['DICAP', 'LEO', '', 0, 0, 0, 0, 0, 1, 'Unknown'],
                ['BRANDO', 'MARLON', 'E', 0, 0, 0, 0, 0, 0, 'White'],
                ['HANKS', 'TOM', '', 0, 0, 0, 0, 0, 1, 'Unknown'],
                ['DENIRO', 'ROBERT', 'E', 0, 1, 0, 0, 0, 1, 'White'],
                ['PACINO', 'AL', 'E', 0, 0, 0, 0, 0, 1, 'White'],
                ['WILLIAMS', 'ROBIN', 'E', 0, 0, 1, 0, 0, 0, 'White'],
                ['EASTWOOD', 'CLINT', 'E', 0, 0, 0, 0, 0, 1, 'White']]

    # convert list of lists into DataFrame
    df = pd.DataFrame(columns=['lname', 'fname', 'rno_cd', 'eri_afr_amer',
                               'eri_asian', 'eri_hawaiian', 'eri_hispanic',
                               'eri_nat_amer', 'eri_white', 'rno_defined'], data=big_list)
    df_nice("30.1", df)
    #
    # ******************
    df.loc[df['eri_white'] == 1, 'race_label'] = 'White'
    df.loc[df['eri_hawaiian'] == 1, 'race_label'] = 'Haw/Pac Isl.'
    df.loc[df['eri_afr_amer'] == 1, 'race_label'] = 'Black/AA'
    df.loc[df['eri_asian'] == 1, 'race_label'] = 'Asian'
    df.loc[df['eri_nat_amer'] == 1, 'race_label'] = 'A/I AK Native'
    df.loc[(df['eri_afr_amer'] + df['eri_asian'] + df['eri_hawaiian'] + df['eri_nat_amer'] + df[
        'eri_white']) > 1, 'race_label'] = 'Two Or More'
    df.loc[df['eri_hispanic'] == 1, 'race_label'] = 'Hispanic'
    df['race_label'].fillna('Other', inplace=True)
    # ******************
    df_nice("30.2", df)


if __name__ == "__main__":
    if 0:
        one_df_join()
        two_df_merge()
        three_df_map()
        four_df_append()
        five_pd_DataFrame()
        six_df_iloc()
        seven_df_iloc()
        eight_np_to_pd_DataFrame()
        nine_df_insert()
        ten_df_assign()
        eleven_df_renameColumns()
        twelve_df_sortColumns()
        thirteen_df_unique()
        fourteen_df_prettyPrint()
        fifteen_mergeJoinConcat_onIndex()
        sixteen_mergeOnMultipleColumns()
        seventeen_systematicSampling()
        eighteen_dropRowsThatSatisfyCondition()
        nineteen_combineColumns()
        twenty_filterRowsThatSatisfyCondition()
        twentyone_groupBy_aggregate()
        twentytwo_readHtmlTables()
        twentythree_crossJoin()
        twentyfour_pivotTable()
        twentyfive_pivot()
        twentysix_df_compareColumns()
        twentyseven_fillColumnWithValues()
        twentyeight_fillColumnWithValues_OnCondition_v1()
        twentynine_fillColumnWithValues_OnCondition_v2()
        thirty_fillColumnWithValues_OnCondition_v3()
    else:
        thirty_fillColumnWithValues_OnCondition_v3()

