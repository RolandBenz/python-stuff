import bs4 as bs
import urllib.request
import pandas as pd
import sys
#from PyQt4.QtGui import QApplication
from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl
#from PyQt4.QtWebKit import QWebPage
from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebPage


def one():
    #
    # open request: read website
    # returns what you see with: view-source:https://pythonprogramming.net/parsememcparseface/
    source = urllib.request.urlopen('https://pythonprogramming.net/parsememcparseface/').read()
    # soup object lets you parse the source with parser: lxml
    soup = bs.BeautifulSoup(source, 'lxml')
    # prints what you see with: view-source:https://pythonprogramming.net/parsememcparseface/
    print('0:\n', soup)
    #
    # print info about website
    # title of the page
    print('1:\n', soup.title)
    # get attributes:
    print('2:\n', soup.title.name)
    # get values:
    print('3:\n', soup.title.string)
    # beginning navigation:
    print('4:\n', soup.title.parent.name)
    # getting specific values:
    print('5:\n', soup.p)
    print('6:\n', soup.find_all('p'))
    #
    for paragraph in soup.find_all('p'):
        print('7:\n', paragraph.string)
        print('8:\n', str(paragraph.text))
    #
    for url in soup.find_all('a'):
        print('9:\n', url.get('href'))
    #
    print('10:\n', soup.get_text())


def two():
    #
    # open request: read website
    # returns what you see with: view-source:https://pythonprogramming.net/parsememcparseface/
    source = urllib.request.urlopen('https://pythonprogramming.net/parsememcparseface/').read()
    # soup object lets you parse the source with parser: lxml
    soup = bs.BeautifulSoup(source, 'lxml')
    # prints what you see with: view-source:https://pythonprogramming.net/parsememcparseface/
    #print('0:\n', soup)
    #
    # navigation bar
    nav = soup.nav
    print('1:\n', nav)
    # finds two nav bars: mobile & laptop
    for url in nav.find_all('a'):
        print('2:\n', url.get('href'))
    # with body object: no head, just body-part
    body = soup.body
    for paragraph in body.find_all('p'):
        print('3:\n', paragraph.text)
    # same without body object: define class; finds all text between div tags
    for div in soup.find_all('div', class_='body'):
        print('4:\n', div.text)


def three():
    #
    # open request: read website
    # returns what you see with: view-source:https://pythonprogramming.net/parsememcparseface/
    source = urllib.request.urlopen('https://pythonprogramming.net/parsememcparseface/').read()
    # soup object lets you parse the source with parser: lxml
    soup = bs.BeautifulSoup(source, 'lxml')
    # prints what you see with: view-source:https://pythonprogramming.net/parsememcparseface/
    # print('0:\n', soup)
    #
    # table; identical to: table = soup.find('table')
    table = soup.table
    print('1:\n', table)
    # table rows: tr
    table_rows = table.find_all('tr')
    print('2:\n', table_rows)
    # table data : td; table header: th
    for tr in table_rows:
        td = tr.find_all('td')
        print('3:\n', td)
        row = [i.text for i in td] #python magic: extract the text from each element
        print('4:\n', row)
    # pandas library: read website; finds all the tables in the website
    dfs = pd.read_html('https://pythonprogramming.net/parsememcparseface/', header=0)
    print('5:\n', dfs)
    for df in dfs:
        print('6:\n', df)
    # new website: parse the sitemap xml-file: maps of all the urls on the website
    source = urllib.request.urlopen('https://pythonprogramming.net/sitemap.xml').read()
    # soup object lets you parse the source with parser: xml
    soup = bs.BeautifulSoup(source, 'xml')
    print('7:\n', soup)
    # just the tag: loc
    for url in soup.find_all('loc'):
        print('8:\n', url.text)


def four():
    #Many websites will supply data that is dynamically loaded via javascript.
    # In Python, you can make use of jinja templating and do this without javascript,
    # but many websites use javascript to populate data.
    #
    # open request: read website
    # returns what you see with: view-source:https://pythonprogramming.net/parsememcparseface/
    source = urllib.request.urlopen('https://pythonprogramming.net/parsememcparseface/').read()
    # soup object lets you parse the source with parser: lxml
    soup = bs.BeautifulSoup(source, 'lxml')
    # prints what you see with: view-source:https://pythonprogramming.net/parsememcparseface/
    # print('0:\n', soup)
    #
    # find p-tag of class jstest; prints: y u bad tho? i.e. text before the .js code was run
    js_test = soup.find('p', class_='jstest')
    print('1:\n', js_test.text)
    #


if __name__ == '__main__':
    #
    if 0:
        one()
        two()
        three()
    else:
        four()