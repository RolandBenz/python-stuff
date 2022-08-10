import dtale
import pandas as pd

"""
Dtale normally used in Jupyter Lab.
But as this script shows, dtale can also be started from within Python.
"""


def one():
    dtale.show(pd.DataFrame([1, 2, 3, 4, 5]), subprocess=False)


def two():
    df = pd.read_csv(r"C:\Users\41792\AppData\Local\pandasgui\dataset_files\diamonds.csv")
    dtale.show(df, subprocess=False)


if __name__ == '__main__':
    if 0:
        one()
        two()
    else:
        two()
