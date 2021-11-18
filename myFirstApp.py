from PyQt5 import QtWidgets, uic
import sys

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('Gui.ui', self) # Load the .ui file
        self.show() # Show the GUI

    def buttonClicked(self):
        print('button clicked!')

app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
app.exec_() # Start the application

# -*- coding: utf-8 -*-

###############################################################################
# Form generated from reading UI file 'GuiRtmuQz.ui'
#
# Created by: Qt User Interface Compiler version 5.14.1
#
# WARNING! All changes made in this file will be lost when recompiling UI file!
###############################################################################