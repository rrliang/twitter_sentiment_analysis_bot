# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'responsewidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ResponseWidget(object):
    def setupUi(self, ResponseWidget):
        ResponseWidget.setObjectName("ResponseWidget")
        ResponseWidget.resize(582, 184)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(ResponseWidget.sizePolicy().hasHeightForWidth())
        ResponseWidget.setSizePolicy(sizePolicy)
        ResponseWidget.setSizeIncrement(QtCore.QSize(0, 1))
        ResponseWidget.setStyleSheet("background-color:rgb(21,32,43);\n"
"border: 1px solid rgb(56,68,77);")
        self.horizontalLayout = QtWidgets.QHBoxLayout(ResponseWidget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(11, 0, 4, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(ResponseWidget)
        self.label_3.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setMinimumSize(QtCore.QSize(61, 61))
        self.label_3.setStyleSheet("background-color:rgb(21,32,43);\n"
"border:none;")
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.label = QtWidgets.QLabel(ResponseWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(481, 131))
        self.label.setSizeIncrement(QtCore.QSize(0, 1))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(-1)
        self.label.setFont(font)
        self.label.setStyleSheet("color:white;\n"
"font-size:20px;\n"
"overflow: hidden;\n"
"padding: 12px;\n"
"position: relative;\n"
"white-space: pre-wrap;\n"
"min-height: 24px;\n"
"max-height: 720px;\n"
"border:none;")
        self.label.setLineWidth(6)
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)

        self.retranslateUi(ResponseWidget)
        QtCore.QMetaObject.connectSlotsByName(ResponseWidget)

    def retranslateUi(self, ResponseWidget):
        _translate = QtCore.QCoreApplication.translate
        ResponseWidget.setWindowTitle(_translate("ResponseWidget", "Form"))
        self.label_3.setText(_translate("ResponseWidget", "<html><head/><body><p><br/></p><p><img src=\"Resources/drislampfp.ico\"/></p></body></html>"))
        self.label.setText(_translate("ResponseWidget", "<html><head/><body><p><span style=\" font-size:20px; color:#ffffff;\">Shiekh Islam</span><span style=\" font-size:20px;\"/><img src=\"Resources/verifiedtwitter.ico\"/><span style=\" font-size:20px;\"/><span style=\" font-size:20px; color:#8899a6;\">@SsHIs </span><span style=\" font-size:20px;\"/></p><p><br/></p></body></html>"))
