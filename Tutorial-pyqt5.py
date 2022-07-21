#! /usr/bin/env python3
#  -*- coding:utf-8 -*-

import sys, time
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QAction, QMessageBox
from PyQt5.QtWidgets import QCheckBox, QProgressBar, QComboBox, QLabel, QStyleFactory, QFontDialog
from PyQt5.QtWidgets import QColorDialog, QCalendarWidget, QTextEdit, QFileDialog


def one():
    #
    # create QApplication; if started from command line and pass arguments: sys.argv holds them
    app = QApplication(sys.argv)
    # create QWidget: window; window-size; window-title
    window = QWidget()
    window.setGeometry(50, 50, 500, 300)
    window.setWindowTitle('pyQt Tuts')
    # show window
    window.show()
    #
    sys.exit(app.exec_())


def two():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = TwoWindow()
    sys.exit(app.exec_())


class TwoWindow(QMainWindow):
    #
    def __init__(self):
        #create QMainWindow(QWidget): window; window-size; window-title
        super(TwoWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        self.show()


def three():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = ThreeWindow()
    sys.exit(app.exec_())


class ThreeWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title
        super(ThreeWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        self.home()

    def home(self):
        # add QPushButton: to quit
        btn = QPushButton('quit', self)
        btn.clicked.connect(QCoreApplication.instance().quit)
        btn.resize(100, 100)
        btn.move(100, 100)
        self.show()


def four():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = FourWindow()
    sys.exit(app.exec_())


class FourWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(FourWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        self.home()

    def home(self):
        # add QPushButton: to quit
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 0)
        self.show()

    def close_application(self):
        print('whooo so custom')
        sys.exit()


def five():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = FiveWindow()
    sys.exit(app.exec_())


class FiveWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(FiveWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction = QAction('&Get to the choppah', self)
        extractAction.setShortcut('Ctrl+Q')
        extractAction.setStatusTip('leave the app')
        # action when triggered / clicked: close app
        extractAction.triggered.connect(self.close_application)
        # call QWindow method: add the status bar
        self.statusBar()
        # call QWindow method: menu bar
        # add menu element with shown text: File; with above defined QAction added
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction)
        # call
        self.home()

    def home(self):
        # add QPushButton: to quit
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        self.show()

    def close_application(self):
        print('whooo so custom')
        sys.exit()


def six():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = SixWindow()
    sys.exit(app.exec_())


class SixWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(SixWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction = QAction('&Get to the choppah', self)
        extractAction.setShortcut('Ctrl+Q')
        extractAction.setStatusTip('leave the app')
        # action when triggered / clicked: close app
        extractAction.triggered.connect(self.close_application)
        # call QMainWindow method: add the status bar
        self.statusBar()
        # call QMainWindow method: menu bar
        # add menu element with shown text: File; with above defined QAction added
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction)
        # QAction: define object with shown icon and text
        extractAction = QAction(QIcon('pic.png'), 'flee the scene', self)
        # action when triggered / clicked: close app
        extractAction.triggered.connect(self.close_application)
        # call QMainWindow method: add toolbar; assign above defined QAction
        self.toolBar = self.addToolBar('Extraction')
        self.toolBar.addAction(extractAction)
        self.home()

    def home(self):
        # add QPushButton: to quit
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        self.show()

    def close_application(self):
        print('whooo so custom')
        sys.exit()


def seven():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = SevenWindow()
    sys.exit(app.exec_())


class SevenWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(SevenWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction = QAction('&Get to the choppah', self)
        extractAction.setShortcut('Ctrl+Q')
        extractAction.setStatusTip('leave the app')
        # action when triggered / clicked: close app
        extractAction.triggered.connect(self.close_application)
        # call QMainWindow method: add the status bar
        self.statusBar()
        # call QMainWindow method: menu bar
        # add menu element with shown text: File; with above defined QAction added
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction)
        # QAction: define object with shown icon and text
        extractAction = QAction(QIcon('pic.png'), 'flee the scene', self)
        # action when triggered / clicked: close app
        extractAction.triggered.connect(self.close_application)
        # call QMainWindow method: add toolbar; assign above defined QAction
        self.toolBar = self.addToolBar('Extraction')
        self.toolBar.addAction(extractAction)
        self.home()

    #
    def home(self):
        # add QPushButton: to quit
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        self.show()

    #
    def close_application(self):
        # Add QMessageBox: with title Message, Text, Yes and No buttons, default No
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        # quit if yes, else ignore
        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass


def eight():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = EightWindow()
    sys.exit(app.exec_())


class EightWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(EightWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction = QAction('&Get to the choppah', self)
        extractAction.setShortcut('Ctrl+Q')
        extractAction.setStatusTip('leave the app')
        # action when triggered / clicked: close app
        extractAction.triggered.connect(self.close_application)
        # call QMainWindow method: add the status bar
        self.statusBar()
        # call QMainWindow method: menu bar
        # add menu element with shown text: File; with above defined QAction added
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction)
        # QAction: define object with shown icon and text
        extractAction = QAction(QIcon('pic.png'), 'flee the scene', self)
        # action when triggered / clicked: close app
        extractAction.triggered.connect(self.close_application)
        # call QMainWindow method: add toolbar; assign above defined QAction
        self.toolBar = self.addToolBar('Extraction')
        self.toolBar.addAction(extractAction)
        self.home()

    #
    def home(self):
        # add QPushButton: with text quit; action
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        # add QCheckBox: text shown; position; action
        checkBox = QCheckBox('Enlarge window', self)
        checkBox.move(0, 70)
        # checkBox.toggle() 'does not force app to enlarge window at start up
        checkBox.stateChanged.connect(self.enlarge_window)
        #
        self.show()

    #
    def enlarge_window(self, state):
        # window size
        if state == Qt.Checked:
            self.setGeometry(50, 50, 1000, 600)
        else:
            self.setGeometry(50, 50, 500, 300)

    #
    def close_application(self):
        # Add QMessageBox: with title Message, Text, Yes and No buttons, default No
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        # quit if yes, else ignore
        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass


def nine():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = NineWindow()
    sys.exit(app.exec_())


class NineWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(NineWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction_1 = QAction('&Get to the choppah', self)
        extractAction_1.setShortcut('Ctrl+Q')
        extractAction_1.setStatusTip('leave the app')
        # action when triggered / clicked: close app
        extractAction_1.triggered.connect(self.close_application)
        # call QMainWindow method: add the status bar
        self.statusBar()
        # call QMainWindow method: menu bar
        # add menu element with shown text: File; with above defined QAction added
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction_1)
        # QAction: define object with shown icon and text
        extractAction_2 = QAction(QIcon('pic.png'), 'flee the scene', self)
        # action when triggered / clicked: close app
        extractAction_2.triggered.connect(self.close_application)
        # call QMainWindow method: add toolbar with name Extraction; assign above defined QAction
        self.toolBar = self.addToolBar('Extraction')
        self.toolBar.addAction(extractAction_2)
        # declare class attributes used later
        self.completed = None
        self.progress = None
        self.btn = None
        #
        self.home()

    #
    def home(self):
        # add QPushButton: with text quit; action
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        # add QCheckBox: text shown; position; action
        checkBox = QCheckBox('Enlarge window', self)
        checkBox.move(0, 70)
        # checkBox.toggle() 'does not force app to enlarge window at start up
        checkBox.stateChanged.connect(self.enlarge_window)
        # add QProgressBar: position and size
        self.progress = QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        # add QPushButton: text shown; position; action
        self.btn = QPushButton('download', self)
        self.btn.move(200, 120)
        self.btn.clicked.connect(self.download)
        #
        self.show()

    #
    def download(self):
        # increment value of progress bar
        self.completed = 0
        while self.completed < 100:
            self.completed += 1
            # add timer to slow down: sleep 0.01 sec
            time.sleep(0.01)
            # input needs to be integer not float
            self.progress.setValue(self.completed)

    #
    def enlarge_window(self, state):
        # window size
        if state == Qt.Checked:
            self.setGeometry(50, 50, 1000, 600)
        else:
            self.setGeometry(50, 50, 500, 300)

    #
    def close_application(self):
        # Add QMessageBox: with title Message, Text, Yes and No buttons, default No
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        # quit if yes, else ignore
        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass


def ten():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = TenWindow()
    sys.exit(app.exec_())


class TenWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(TenWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction_1 = QAction('&Get to the choppah', self)
        extractAction_1.setShortcut('Ctrl+Q')
        extractAction_1.setStatusTip('leave the app')
        # action when triggered / clicked: close app
        extractAction_1.triggered.connect(self.close_application)
        # call QMainWindow method: add the status bar
        self.statusBar()
        # call QMainWindow method: menu bar
        # add menu element with shown text: File; with above defined QAction added
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction_1)
        # QAction: define object with shown icon and text
        extractAction_2 = QAction(QIcon('pic.png'), 'flee the scene', self)
        # action when triggered / clicked: close app
        extractAction_2.triggered.connect(self.close_application)
        # call QMainWindow method: add toolbar with name Extraction; assign above defined QAction
        self.toolBar = self.addToolBar('Extraction')
        self.toolBar.addAction(extractAction_2)
        # declare class attributes used later
        self.completed = None
        self.progress = None
        self.btn = None
        self.labelStyleChoice = None
        #
        self.home()

    #
    def home(self):
        # add QPushButton: with text quit; action
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        # add QCheckBox: text shown; position; action
        checkBox = QCheckBox('Enlarge window', self)
        checkBox.move(0, 70)
        # checkBox.toggle() 'does not force app to enlarge window at start up
        checkBox.stateChanged.connect(self.enlarge_window)
        # add QProgressBar: position and size
        self.progress = QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        # add QPushButton: text shown; position; action
        self.btn = QPushButton('download', self)
        self.btn.move(200, 120)
        self.btn.clicked.connect(self.download)
        # add QLabel: text shown Windows
        self.labelStyleChoice = QLabel('windowsvista', self)
        self.labelStyleChoice.move(25, 150)
        # add QComboBox: add items to QComboBox; action
        comboBox = QComboBox(self)
        comboBox.addItem('windowsvista')
        comboBox.addItem('Windows')
        comboBox.addItem('Fusion')
        comboBox.move(25, 250)
        comboBox.activated[str].connect(self.style_choice)
        #
        self.show()

    #
    def style_choice(self, text):
        # labelStyleChoice is the QLabel: set new text, chosen in QComboBox
        self.labelStyleChoice.setText(text)
        # set style of app
        QApplication.setStyle(QStyleFactory.create(text))
        # print: available styles; current style
        print(QStyleFactory.keys())
        print(self.style().objectName())

    #
    def download(self):
        # increment value of progress bar
        self.completed = 0
        while self.completed < 100:
            self.completed += 1
            # add timer to slow down: sleep 0.01 sec
            time.sleep(0.01)
            # input needs to be integer not float
            self.progress.setValue(self.completed)

    #
    def enlarge_window(self, state):
        # window size
        if state == Qt.Checked:
            self.setGeometry(50, 50, 1000, 600)
        else:
            self.setGeometry(50, 50, 500, 300)

    #
    def close_application(self):
        # Add QMessageBox: with title Message, Text, Yes and No buttons, default No
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        # quit if yes, else ignore
        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass


def eleven():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = ElevenWindow()
    sys.exit(app.exec_())


class ElevenWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(ElevenWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction_1 = QAction('&Get to the choppah', self)
        extractAction_1.setShortcut('Ctrl+Q')
        extractAction_1.setStatusTip('leave the app')
        # action when triggered / clicked: close app
        extractAction_1.triggered.connect(self.close_application)
        # call QMainWindow method: add the status bar
        self.statusBar()
        # call QMainWindow method: menu bar
        # add menu element with shown text: File; with above defined QAction added
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction_1)
        # QAction: define object with shown icon and text
        extractAction_2 = QAction(QIcon('pic.png'), 'flee the scene', self)
        # action when triggered / clicked: call own method to close app
        extractAction_2.triggered.connect(self.close_application)
        # call QMainWindow method: add toolbar with name Extraction_Toolbar; assign above defined QAction
        self.toolBar = self.addToolBar('Extraction_Toolbar')
        self.toolBar.addAction(extractAction_2)
        # QAction: define object with shown text; when triggered call own method font_choice
        fontChoiceAction = QAction('Font', self)
        fontChoiceAction.triggered.connect(self.font_choice)
        # call QMainWindow method: add toolbar with name Font_Toolbar; assign above defined QAction
        self.toolBar = self.addToolBar('Font_Toolbar')
        self.toolBar.addAction(fontChoiceAction)
        #
        # declare class attributes used later
        self.completed = None
        self.progress = None
        self.btn = None
        self.labelStyleChoice = None
        #
        self.home()

    #
    def font_choice(self):
        # QFontDialog: opens Font Picker
        font, valid = QFontDialog.getFont()
        if valid:
            self.labelStyleChoice.setFont(font)

    #
    def home(self):
        # add QPushButton: with text quit; action
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        # add QCheckBox: text shown; position; action
        checkBox = QCheckBox('Enlarge window', self)
        checkBox.move(0, 70)
        # checkBox.toggle() 'does not force app to enlarge window at start up
        checkBox.stateChanged.connect(self.enlarge_window)
        # add QProgressBar: position and size
        self.progress = QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        # add QPushButton: text shown; position; action
        self.btn = QPushButton('download', self)
        self.btn.move(200, 120)
        self.btn.clicked.connect(self.download)
        # add QLabel: text shown Windows
        self.labelStyleChoice = QLabel('windowsvista', self)
        self.labelStyleChoice.move(25, 150)
        # add QComboBox: add items to QComboBox; action
        comboBox = QComboBox(self)
        comboBox.addItem('windowsvista')
        comboBox.addItem('Windows')
        comboBox.addItem('Fusion')
        comboBox.move(25, 250)
        comboBox.activated[str].connect(self.style_choice)
        #
        self.show()

    #
    def style_choice(self, text):
        # labelStyleChoice is the QLabel: set new text, chosen in QComboBox
        self.labelStyleChoice.setText(text)
        # set style of app
        QApplication.setStyle(QStyleFactory.create(text))
        # print: available styles; current style
        print(QStyleFactory.keys())
        print(self.style().objectName())

    #
    def download(self):
        # increment value of progress bar
        self.completed = 0
        while self.completed < 100:
            self.completed += 1
            # add timer to slow down: sleep 0.01 sec
            time.sleep(0.01)
            # input needs to be integer not float
            self.progress.setValue(self.completed)

    #
    def enlarge_window(self, state):
        # window size
        if state == Qt.Checked:
            self.setGeometry(50, 50, 1000, 600)
        else:
            self.setGeometry(50, 50, 500, 300)

    #
    def close_application(self):
        # Add QMessageBox: with title Message, Text, Yes and No buttons, default No
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        # quit if yes, else ignore
        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass


def twelve():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = TwelveWindow()
    sys.exit(app.exec_())


class TwelveWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(TwelveWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction_1 = QAction('&Get to the choppah', self)
        extractAction_1.setShortcut('Ctrl+Q')
        extractAction_1.setStatusTip('leave the app')
        # action when triggered: close app
        extractAction_1.triggered.connect(self.close_application)
        # call QMainWindow method: add the status bar
        self.statusBar()
        # call QMainWindow method: menu bar
        mainMenu = self.menuBar()
        # add menu element with shown text: File; with above defined QAction added
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction_1)
        # QAction: define object with shown icon and text
        extractAction_2 = QAction(QIcon('pic.png'), 'flee the scene', self)
        # action when triggered / clicked: call own method to close app
        extractAction_2.triggered.connect(self.close_application)
        # call QMainWindow method: add toolbar with name Extraction_Toolbar; assign above defined QAction
        self.toolBar_1 = self.addToolBar('Extraction_Toolbar')
        self.toolBar_1.addAction(extractAction_2)
        # QAction: define object with shown text; when triggered call own method font_choice
        fontChoiceAction = QAction('Font', self)
        fontChoiceAction.triggered.connect(self.font_choice)
        # call QMainWindow method: add toolbar with name Font_Toolbar; assign above defined QAction
        self.toolBar_2 = self.addToolBar('Font_Toolbar')
        self.toolBar_2.addAction(fontChoiceAction)
        # add QCalendarWidget
        cal = QCalendarWidget(self)
        cal.move(200, 200)
        cal.resize(500, 300)
        print(cal.selectedDate())
        #
        # declare class attributes used later
        self.completed = None
        self.progress = None
        self.btn = None
        self.labelStyleChoice = None
        self.textEdit = None
        #
        self.home()

    #
    def font_choice(self):
        # QFontDialog: opens Font Picker
        font, valid = QFontDialog.getFont()
        if valid:
            self.labelStyleChoice.setFont(font)

    #
    def home(self):
        # add QPushButton: with text quit; action
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        # add QCheckBox: text shown; position; action
        checkBox = QCheckBox('Enlarge window', self)
        checkBox.move(0, 70)
        # checkBox.toggle() 'does not force app to enlarge window at start up
        checkBox.stateChanged.connect(self.enlarge_window)
        # add QProgressBar: position and size
        self.progress = QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        # add QPushButton: text shown; position; action
        self.btn = QPushButton('download', self)
        self.btn.move(200, 120)
        self.btn.clicked.connect(self.download)
        # add QLabel: text shown Windows
        self.labelStyleChoice = QLabel('windowsvista', self)
        self.labelStyleChoice.move(25, 150)
        # add QComboBox: add items to QComboBox; action
        comboBox = QComboBox(self)
        comboBox.addItem('windowsvista')
        comboBox.addItem('Windows')
        comboBox.addItem('Fusion')
        comboBox.move(25, 250)
        comboBox.activated[str].connect(self.style_choice)
        # add QAction
        fontColorAction = QAction('font bg color', self)
        fontColorAction.triggered.connect(self.color_picker)
        # add action to toolbar
        self.toolBar_2.addAction(fontColorAction)
        #
        self.show()

    #
    def color_picker(self):
        # QColorDialog: color picker
        color = QColorDialog.getColor()
        # setStyleSheet: change label background-color
        self.labelStyleChoice.setStyleSheet(
            'QWidget{color: blue; border: 1px solid; border-color:red; background-color: %s}' % color.name())

    #
    def style_choice(self, text):
        # labelStyleChoice is the QLabel: set new text, chosen in QComboBox
        self.labelStyleChoice.setText(text)
        # set style of app
        QApplication.setStyle(QStyleFactory.create(text))
        # print: available styles; current style
        print(QStyleFactory.keys())
        print(self.style().objectName())

    #
    def download(self):
        # increment value of progress bar
        self.completed = 0
        while self.completed < 100:
            self.completed += 1
            # add timer to slow down: sleep 0.01 sec
            time.sleep(0.01)
            # input needs to be integer not float
            self.progress.setValue(self.completed)

    #
    def enlarge_window(self, state):
        # window size
        if state == Qt.Checked:
            self.setGeometry(50, 50, 1000, 600)
        else:
            self.setGeometry(50, 50, 500, 300)

    #
    def close_application(self):
        # Add QMessageBox: with title Message, Text, Yes and No buttons, default No
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        # quit if yes, else ignore
        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass


def thirteen():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = ThirteenWindow()
    sys.exit(app.exec_())


class ThirteenWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(ThirteenWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction_1 = QAction('&Get to the choppah', self)
        extractAction_1.setShortcut('Ctrl+Q')
        extractAction_1.setStatusTip('leave the app')
        # action when triggered: close app
        extractAction_1.triggered.connect(self.close_application)
        # QAction: define object with shown text; shortcut; status bar tip
        openEditorAction = QAction('&Editor', self)
        openEditorAction.setShortcut('Ctrl+E')
        openEditorAction.setStatusTip('Open Editor')
        # action when triggered: open editor
        openEditorAction.triggered.connect(self.open_editor)
        # call QMainWindow method: add the status bar
        self.statusBar()
        # call QMainWindow method: menu bar
        mainMenu = self.menuBar()
        # add menu element with shown text: File; with above defined QAction added
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction_1)
        # add menu element with shown text: Editor, with above defined QAction added
        editorMenu = mainMenu.addMenu('&Editor')
        editorMenu.addAction(openEditorAction)
        # QAction: define object with shown icon and text
        extractAction_2 = QAction(QIcon('pic.png'), 'flee the scene', self)
        # action when triggered / clicked: call own method to close app
        extractAction_2.triggered.connect(self.close_application)
        # call QMainWindow method: add toolbar with name Extraction_Toolbar; assign above defined QAction
        self.toolBar_1 = self.addToolBar('Extraction_Toolbar')
        self.toolBar_1.addAction(extractAction_2)
        # QAction: define object with shown text; when triggered call own method font_choice
        fontChoiceAction = QAction('Font', self)
        fontChoiceAction.triggered.connect(self.font_choice)
        # call QMainWindow method: add toolbar with name Font_Toolbar; assign above defined QAction
        self.toolBar_2 = self.addToolBar('Font_Toolbar')
        self.toolBar_2.addAction(fontChoiceAction)
        # add QCalendarWidget
        cal = QCalendarWidget(self)
        cal.move(200, 200)
        cal.resize(500, 300)
        print(cal.selectedDate())
        #
        # declare class attributes used later
        self.completed = None
        self.progress = None
        self.btn = None
        self.labelStyleChoice = None
        self.textEdit = None
        #
        self.home()

    #
    def open_editor(self):
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)

    #
    def font_choice(self):
        # QFontDialog: opens Font Picker
        font, valid = QFontDialog.getFont()
        if valid:
            self.labelStyleChoice.setFont(font)

    #
    def home(self):
        # add QPushButton: with text quit; action
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        # add QCheckBox: text shown; position; action
        checkBox = QCheckBox('Enlarge window', self)
        checkBox.move(0, 70)
        # checkBox.toggle() 'does not force app to enlarge window at start up
        checkBox.stateChanged.connect(self.enlarge_window)
        # add QProgressBar: position and size
        self.progress = QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        # add QPushButton: text shown; position; action
        self.btn = QPushButton('download', self)
        self.btn.move(200, 120)
        self.btn.clicked.connect(self.download)
        # add QLabel: text shown Windows
        self.labelStyleChoice = QLabel('windowsvista', self)
        self.labelStyleChoice.move(25, 150)
        # add QComboBox: add items to QComboBox; action
        comboBox = QComboBox(self)
        comboBox.addItem('windowsvista')
        comboBox.addItem('Windows')
        comboBox.addItem('Fusion')
        comboBox.move(25, 250)
        comboBox.activated[str].connect(self.style_choice)
        # add QAction
        fontColorAction = QAction('font bg color', self)
        fontColorAction.triggered.connect(self.color_picker)
        # add action to toolbar
        self.toolBar_2.addAction(fontColorAction)
        #
        self.show()

    #
    def color_picker(self):
        # QColorDialog: color picker
        color = QColorDialog.getColor()
        # setStyleSheet: change label background-color
        self.labelStyleChoice.setStyleSheet(
            'QWidget{color: blue; border: 1px solid; border-color:red; background-color: %s}' % color.name())

    #
    def style_choice(self, text):
        # labelStyleChoice is the QLabel: set new text, chosen in QComboBox
        self.labelStyleChoice.setText(text)
        # set style of app
        QApplication.setStyle(QStyleFactory.create(text))
        # print: available styles; current style
        print(QStyleFactory.keys())
        print(self.style().objectName())

    #
    def download(self):
        # increment value of progress bar
        self.completed = 0
        while self.completed < 100:
            self.completed += 1
            # add timer to slow down: sleep 0.01 sec
            time.sleep(0.01)
            # input needs to be integer not float
            self.progress.setValue(self.completed)

    #
    def enlarge_window(self, state):
        # window size
        if state == Qt.Checked:
            self.setGeometry(50, 50, 1000, 600)
        else:
            self.setGeometry(50, 50, 500, 300)

    #
    def close_application(self):
        # Add QMessageBox: with title Message, Text, Yes and No buttons, default No
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        # quit if yes, else ignore
        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass


def fourteen():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = FourteenWindow()
    sys.exit(app.exec_())


class FourteenWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(FourteenWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction_1 = QAction('&Get to the choppah', self)
        extractAction_1.setShortcut('Ctrl+Q')
        extractAction_1.setStatusTip('leave the app')
        # action when triggered: close app
        extractAction_1.triggered.connect(self.close_application)
        # QAction: define object with shown text; shortcut; status bar tip
        openEditorAction = QAction('&Editor', self)
        openEditorAction.setShortcut('Ctrl+E')
        openEditorAction.setStatusTip('Open Editor')
        # action when triggered: open editor
        openEditorAction.triggered.connect(self.open_editor)
        # QAction: define object with shown text; shortcut; status bar tip
        openFileAction = QAction('&Open File', self)
        openFileAction.setShortcut('Ctrl+O')
        openFileAction.setStatusTip('Open File')
        # action when triggered: open editor
        openFileAction.triggered.connect(self.file_open)
        # call QMainWindow method: add the status bar
        self.statusBar()
        # call QMainWindow method: menu bar
        mainMenu = self.menuBar()
        # add menu element with shown text: File
        fileMenu = mainMenu.addMenu('&File')
        # add QActions for: extractAction_1 and openFileAction; adds this to menu File
        fileMenu.addAction(extractAction_1)
        fileMenu.addAction(openFileAction)
        # add menu element with shown text: Editor, with above defined QAction added
        editorMenu = mainMenu.addMenu('&Editor')
        editorMenu.addAction(openEditorAction)
        # QAction: define object with shown icon and text
        extractAction_2 = QAction(QIcon('pic.png'), 'flee the scene', self)
        # action when triggered / clicked: call own method to close app
        extractAction_2.triggered.connect(self.close_application)
        # call QMainWindow method: add toolbar with name Extraction_Toolbar; assign above defined QAction
        self.toolBar_1 = self.addToolBar('Extraction_Toolbar')
        self.toolBar_1.addAction(extractAction_2)
        # QAction: define object with shown text; when triggered call own method font_choice
        fontChoiceAction = QAction('Font', self)
        fontChoiceAction.triggered.connect(self.font_choice)
        # call QMainWindow method: add toolbar with name Font_Toolbar; assign above defined QAction
        self.toolBar_2 = self.addToolBar('Font_Toolbar')
        self.toolBar_2.addAction(fontChoiceAction)
        # add QCalendarWidget
        cal = QCalendarWidget(self)
        cal.move(200, 200)
        cal.resize(500, 300)
        print(cal.selectedDate())
        #
        # declare class attributes used later
        self.completed = None
        self.progress = None
        self.btn = None
        self.labelStyleChoice = None
        self.textEditor = None
        #
        self.home()

    #
    def file_open(self):
        # QFileDialog: file picker
        name, _ = QFileDialog.getOpenFileName(
            self, 'Open File', options=QFileDialog.DontUseNativeDialog)
        # built in method open: open the file
        file = open(name, 'r')
        # call own method: open editor
        self.open_editor()
        with file:
            # read file
            text = file.read()
            # put text into text editor
            self.textEditor.setText(text)

    #
    def open_editor(self):
        # QTextEdit: text editor
        self.textEditor = QTextEdit()
        # allows to take up the entire application
        self.setCentralWidget(self.textEditor)

    #
    def font_choice(self):
        # QFontDialog: opens Font Picker
        font, valid = QFontDialog.getFont()
        if valid:
            self.labelStyleChoice.setFont(font)

    #
    def home(self):
        # add QPushButton: with text quit; action
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        # add QCheckBox: text shown; position; action
        checkBox = QCheckBox('Enlarge window', self)
        checkBox.move(0, 70)
        # checkBox.toggle() 'does not force app to enlarge window at start up
        checkBox.stateChanged.connect(self.enlarge_window)
        # add QProgressBar: position and size
        self.progress = QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        # add QPushButton: text shown; position; action
        self.btn = QPushButton('download', self)
        self.btn.move(200, 120)
        self.btn.clicked.connect(self.download)
        # add QLabel: text shown Windows
        self.labelStyleChoice = QLabel('windowsvista', self)
        self.labelStyleChoice.move(25, 150)
        # add QComboBox: add items to QComboBox; action
        comboBox = QComboBox(self)
        comboBox.addItem('windowsvista')
        comboBox.addItem('Windows')
        comboBox.addItem('Fusion')
        comboBox.move(25, 250)
        comboBox.activated[str].connect(self.style_choice)
        # add QAction
        fontColorAction = QAction('font bg color', self)
        fontColorAction.triggered.connect(self.color_picker)
        # add action to toolbar
        self.toolBar_2.addAction(fontColorAction)
        #
        self.show()

    #
    def color_picker(self):
        # QColorDialog: color picker
        color = QColorDialog.getColor()
        # setStyleSheet: change label background-color
        self.labelStyleChoice.setStyleSheet(
            'QWidget{color: blue; border: 1px solid; border-color:red; background-color: %s}' % color.name())

    #
    def style_choice(self, text):
        # labelStyleChoice is the QLabel: set new text, chosen in QComboBox
        self.labelStyleChoice.setText(text)
        # set style of app
        QApplication.setStyle(QStyleFactory.create(text))
        # print: available styles; current style
        print(QStyleFactory.keys())
        print(self.style().objectName())

    #
    def download(self):
        # increment value of progress bar
        self.completed = 0
        while self.completed < 100:
            self.completed += 1
            # add timer to slow down: sleep 0.01 sec
            time.sleep(0.01)
            # input needs to be integer not float
            self.progress.setValue(self.completed)

    #
    def enlarge_window(self, state):
        # window size
        if state == Qt.Checked:
            self.setGeometry(50, 50, 1000, 600)
        else:
            self.setGeometry(50, 50, 500, 300)

    #
    def close_application(self):
        # Add QMessageBox: with title Message, Text, Yes and No buttons, default No
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        # quit if yes, else ignore
        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass


def fifteen():
    #
    # create QApplication
    app = QApplication(sys.argv)
    Gui = FifteenWindow()
    sys.exit(app.exec_())


class FifteenWindow(QMainWindow):
    #
    def __init__(self):
        # create QMainWindow(QWidget): window; window-size; window-title; icon
        super(FifteenWindow, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        self.setWindowIcon(QIcon('pic.png'))
        # QAction: define object with shown text; shortcut; status bar tip
        extractAction_1 = QAction('&Get to the choppah', self)
        extractAction_1.setShortcut('Ctrl+Q')
        extractAction_1.setStatusTip('leave the app')
        # action when triggered: close app
        extractAction_1.triggered.connect(self.close_application)
        # QAction: define object with shown text; shortcut; status bar tip
        openEditorAction = QAction('&Editor', self)
        openEditorAction.setShortcut('Ctrl+E')
        openEditorAction.setStatusTip('Open Editor')
        # action when triggered: open editor
        openEditorAction.triggered.connect(self.open_editor)
        # QAction: define object with shown text; shortcut; status bar tip
        openFileAction = QAction('&Open File', self)
        openFileAction.setShortcut('Ctrl+O')
        openFileAction.setStatusTip('Open File')
        # action when triggered: open editor
        openFileAction.triggered.connect(self.file_open)
        # QAction: define object with shown text; shortcut; status bar tip
        saveFileAction = QAction('&Save File', self)
        saveFileAction.setShortcut('Ctrl+S')
        saveFileAction.setStatusTip('Save File')
        # action when triggered: open editor
        saveFileAction.triggered.connect(self.file_save)
        # call QMainWindow method: add the status bar
        self.statusBar()
        # call QMainWindow method: menu bar
        mainMenu = self.menuBar()
        # add menu element with shown text: File
        fileMenu = mainMenu.addMenu('&File')
        # add QActions for: extractAction_1, openFileAction, saveFileAction; adds this to menu File
        fileMenu.addAction(extractAction_1)
        fileMenu.addAction(openFileAction)
        fileMenu.addAction(saveFileAction)
        # add menu element with shown text: Editor, with above defined QAction added
        editorMenu = mainMenu.addMenu('&Editor')
        editorMenu.addAction(openEditorAction)
        # QAction: define object with shown icon and text
        extractAction_2 = QAction(QIcon('pic.png'), 'flee the scene', self)
        # action when triggered / clicked: call own method to close app
        extractAction_2.triggered.connect(self.close_application)
        # call QMainWindow method: add toolbar with name Extraction_Toolbar; assign above defined QAction
        self.toolBar_1 = self.addToolBar('Extraction_Toolbar')
        self.toolBar_1.addAction(extractAction_2)
        # QAction: define object with shown text; when triggered call own method font_choice
        fontChoiceAction = QAction('Font', self)
        fontChoiceAction.triggered.connect(self.font_choice)
        # call QMainWindow method: add toolbar with name Font_Toolbar; assign above defined QAction
        self.toolBar_2 = self.addToolBar('Font_Toolbar')
        self.toolBar_2.addAction(fontChoiceAction)
        # add QCalendarWidget
        cal = QCalendarWidget(self)
        cal.move(200, 200)
        cal.resize(500, 300)
        print(cal.selectedDate())
        #
        # declare class attributes used later
        self.completed = None
        self.progress = None
        self.btn = None
        self.labelStyleChoice = None
        self.textEditor = None
        #
        self.home()

    #
    def file_save(self):
        # QFileDialog: save file dialog
        name, _ = QFileDialog.getSaveFileName(
            self,'Save File', options=QFileDialog.DontUseNativeDialog)
        # built in method open: open the file in w mode
        file = open(name, 'w')
        # take the text from the editor and write it to the file, close it
        text = self.textEditor.toPlainText()
        file.write(text)
        file.close()

    #
    def file_open(self):
        # QFileDialog: file picker
        name, _ = QFileDialog.getOpenFileName(
            self, 'Open File', options=QFileDialog.DontUseNativeDialog)
        # built in method open: open the file in r mode
        file = open(name, 'r')
        # call own method: open editor
        self.open_editor()
        with file:
            # read file
            text = file.read()
            # put text into text editor
            self.textEditor.setText(text)

    #
    def open_editor(self):
        # QTextEdit: text editor
        self.textEditor = QTextEdit()
        # allows to take up the entire application
        self.setCentralWidget(self.textEditor)

    #
    def font_choice(self):
        # QFontDialog: opens Font Picker
        font, valid = QFontDialog.getFont()
        if valid:
            self.labelStyleChoice.setFont(font)

    #
    def home(self):
        # add QPushButton: with text quit; action
        btn = QPushButton('quit', self)
        btn.clicked.connect(self.close_application)
        # set to acceptable size automatic; location; show
        btn.resize(btn.sizeHint())
        btn.move(0, 100)
        # add QCheckBox: text shown; position; action
        checkBox = QCheckBox('Enlarge window', self)
        checkBox.move(0, 70)
        # checkBox.toggle() 'does not force app to enlarge window at start up
        checkBox.stateChanged.connect(self.enlarge_window)
        # add QProgressBar: position and size
        self.progress = QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        # add QPushButton: text shown; position; action
        self.btn = QPushButton('download', self)
        self.btn.move(200, 120)
        self.btn.clicked.connect(self.download)
        # add QLabel: text shown Windows
        self.labelStyleChoice = QLabel('windowsvista', self)
        self.labelStyleChoice.move(25, 150)
        # add QComboBox: add items to QComboBox; action
        comboBox = QComboBox(self)
        comboBox.addItem('windowsvista')
        comboBox.addItem('Windows')
        comboBox.addItem('Fusion')
        comboBox.move(25, 250)
        comboBox.activated[str].connect(self.style_choice)
        # add QAction
        fontColorAction = QAction('font bg color', self)
        fontColorAction.triggered.connect(self.color_picker)
        # add action to toolbar
        self.toolBar_2.addAction(fontColorAction)
        #
        self.show()

    #
    def color_picker(self):
        # QColorDialog: color picker
        color = QColorDialog.getColor()
        # setStyleSheet: change label background-color
        self.labelStyleChoice.setStyleSheet(
            'QWidget{color: blue; border: 1px solid; border-color:red; background-color: %s}' % color.name())

    #
    def style_choice(self, text):
        # labelStyleChoice is the QLabel: set new text, chosen in QComboBox
        self.labelStyleChoice.setText(text)
        # set style of app
        QApplication.setStyle(QStyleFactory.create(text))
        # print: available styles; current style
        print(QStyleFactory.keys())
        print(self.style().objectName())

    #
    def download(self):
        # increment value of progress bar
        self.completed = 0
        while self.completed < 100:
            self.completed += 1
            # add timer to slow down: sleep 0.01 sec
            time.sleep(0.01)
            # input needs to be integer not float
            self.progress.setValue(self.completed)

    #
    def enlarge_window(self, state):
        # window size
        if state == Qt.Checked:
            self.setGeometry(50, 50, 1000, 600)
        else:
            self.setGeometry(50, 50, 500, 300)

    #
    def close_application(self):
        # Add QMessageBox: with title Message, Text, Yes and No buttons, default No
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        # quit if yes, else ignore
        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass


if __name__ == '__main__':
    if 0:
        one()
        two()
        three()
        four()
        five()
        six()
        seven()
        eight()
        nine()
        ten()
        eleven()
        twelve()
        thirteen()
        fourteen()
    else:
        fifteen()
