"""
I only have Office / Excel in read only view.
Therefore, I could not test the code.
"""

import pandas as pd
import xlwings as xw

def one():
    """
    download .xlsx from url with pandas
    view in Excel with xlwings
    """
    url = 'https://github.com/chris1610/pbpython/blob/master/data/2018_Sales_Total_v2.xlsx?raw=True'
    df = pd.read_excel(url)
    # Create a new workbook and add the DataFrame to Sheet1
    xw.view(df)

def two():
    """
    open empty .xlsx with xlwings
    add a new sheet
    """
    book = xw.Book()
    sheet1 = book.sheets('Sheet1')
    sheet2 = book.sheets('Sheet2')

def three():
    """
    open .xlsx from path with xlwings
    add sheets and change content
    """
    book = xw.Book(r'data/2018_Sales_Total_v2.xlsx')
    sheet1 = book.sheets('Sheet1')
    sheet2 = book.sheets('Sheet2')
    sheet3 = book.sheets('Sheet3')
    sheet4 = book.sheets('Sheet4')
    sheet5 = book.sheets('Sheet5')
    sheet6 = book.sheets('Sheet5')
    sheet7 = book.sheets('Sheet5')
    sheet8 = book.sheets('Sheet5')
    """
    # ranges
    """
    print(sheet1.range('A1'))
    print(sheet1.range('A1:C3'))
    print(sheet1.range((1, 1)))
    print(sheet1.range((1, 1), (3, 3)))
    print(sheet1.range(xw.Range('A1'), sheet1.range('B2')))
    #
    rng = sheet1.range('A1:D5')
    print(rng[0, 0])  # <Range [Workbook1]Sheet1!$A$1>
    print(rng[1])  # <Range [Workbook1]Sheet1!$B$1>
    print(rng[:, 3:])  # <Range [Workbook1]Sheet1!$D$1:$D$5>
    print(rng[1:3, 1:3])  # <Range [Book1]Sheet1!$A$1:$J$10>

    """
    # single cells
    """
    sheet2.range('A1').value = 1
    sheet2.range('A2').value = 'Hello'
    sheet2.range('A4').value = dt.datetime(2000, 1, 1)
    print(sheet2.range('A1').value)
    print(sheet2.range('A2').value)
    print(sheet2.range('A3').value is None)
    print(sheet2.range('A4').value)

    """
    lists
    """
    sheet3.range('A1').value = [[1], [2], [3], [4], [5]]  # Column orientation (nested list)
    sheet3.range('A1').value = [1, 2, 3, 4, 5]
    print(sheet3.range('A1:A5').value)
    print(sheet3.range('A1:E1').value)
    # To force a single cell to arrive as list, use:
    print(sheet3.range('A1').options(ndim=1).value)
    # 2d lists: If the row or column orientation has to be preserved, set ndim in the Range options.
    # This will return the Ranges as nested lists (“2d lists”):
    print(sheet3.range('A1:A5').options(ndim=2).value)
    print(sheet3.range('A1:E1').options(ndim=2).value)
    # 2 dimensional Ranges are automatically returned as nested lists.
    # When assigning (nested) lists to a Range in Excel,
    # it’s enough to just specify the top left cell as target address.
    # This sample also makes use of index notation to read the values back into Python:
    sheet3.range('A10').value = [['Foo 1', 'Foo 2', 'Foo 3'], [10, 20, 30]]
    print(sheet3.range((10, 1), (11, 3)).value)

    """
    # range expanding
    # While expand gives back an expanded Range object, 
    # options are only evaluated when accessing the values of a Range.
    """
    sheet4.range('A1').value = [[1, 2], [3, 4]]
    rng1 = sheet4.range('A1').expand('table')  # or just .expand()
    rng2 = sheet4.range('A1').options(expand='table')
    print(rng1.value)
    print(rng2.value)
    #
    sheet4.range('A3').value = [5, 6]
    print(rng1.value)
    print(rng2.value)

    """
    # numpy arrays
    """
    sheet5.range('A1').value = np.eye(3)
    print(sheet5.range('A1').options(np.array, expand='table').value)

    """
    # pandas df
    """
    df = pd.DataFrame([[1.1, 2.2], [3.3, None]], columns=['one', 'two'])
    sheet6.range('A1').value = df
    print(sheet.range('A1:C3').options(pd.DataFrame).value)
    # options: work for reading and writing
    sheet6.range('A5').options(index=False).value = df
    sheet6.range('A9').options(index=False, header=False).value = df

    """
    # pandas series
    """
    s = pd.Series([1.1, 3.3, 5., np.nan, 6., 8.], name='myseries')
    sheet7.range('A1').value = s
    print(sheet7.range('A1:B7').options(pd.Series).value)

    """
    # chunking big datasets
    """
    # For writing
    data = np.arange(75_000 * 20).reshape(75_000, 20)
    df1 = pd.DataFrame(data=data)
    sheet8.options(chunksize=10_000).value = df1
    print(data.shape)

    # And the same for reading:
    # As DataFrame
    df2 = sheet8.expand().options(pd.DataFrame, chunksize=10_000).value
    print(df2.info())
    # As list of list
    df3 = sheet8.expand().options(chunksize=10_000).value
    print(df3.info())


if __name__ == '__main__':
    if 0:
        one()
        two()
        three()
    else:
        one()