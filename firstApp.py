from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QListWidgetItem, QFrame, QHBoxLayout, QAction
from gui import *
from responsewidget import*
from PyQt5 import QtWidgets
import re
import pickle
from spacy.lang.en import STOP_WORDS
import sklearn
import nltk
import sys

with open('model_pickle','rb') as f:
   model = pickle.load(f)
with open('cv','rb') as f:
   cv = pickle.load(f)
def preprocess_string(stringtopre):
   stringtopre = re.sub(r'http?:\/\/\S+', '', stringtopre)  # Remove links with https
   stringtopre = re.sub(r"www\.[a-z]?\.?@[\w]+(com)+|[a-z]+\.(com)", '', stringtopre)  # Remove links with www. and com
   stringtopre = stringtopre.lower()  # Lower case all tweets
   stringtopre = re.sub('@[^\s]+', '', stringtopre)  # Remove @username
   stringtopre = re.sub("#[A-Za-z0-9_]", '', stringtopre)  # Remove #hashtag
   stringtopre = re.sub(' RT ', "", stringtopre)  # Remove RT (Retweet)

   contractions = {
       " aight ": " alright ",
       " ain't ": " am not ",
       " amn't ": " am not ",
       " aren't ": " are not ",
       " can't ": " can not ",
       "cant ": " can not ",
       " cause ": " because ",
       " could've ": " could have ",
       " couldn't ": " could not ",
       " couldn't've ": " could not have ",
       " daren't ": " dare not ",
       " daresn't ": " dare not ",
       " dasn't ": " dare not ",
       " didn't ": " did not ",
       " doesn't": " does not ",
       " don't ": " do not ",
       " d'oh ": " doh ",
       " d'ye ": " do you ",
       " e'er ": " ever ",
       " everybody's ": " everybody is ",
       " everyone's ": " everyone is ",
       " finna ": " fixing to ",
       " g'day ": " good day ",
       " gimme ": " give me ",
       " giv'n ": " given ",
       " gonna ": " going to ",
       " gon't ": " go not ",
       " gotta ": " got to ",
       " hadn't ": " had not ",
       " had've ": " had have ",
       " hasn't ": " has not ",
       " haven't ": " have not ",
       " havent ": " have not ",
       " he'd ": " he had ",
       " he'dn't've'd ": " he would not have had ",
       " he'll ": " he will ",
       " he's ": " he is ",
       " he've ": " he have ",
       " how'd ": " how would ",
       " howdy ": " how do you do ",
       " how'll ": " how will ",
       " how're ": " how are ",
       " i'll ": " i will ",
       " im ": " i am ",
       " im ": " i am",
       " i'm ": " i am ",
       " i'm'a": " i am about to ",
       " i'm'o ": " i am going to ",
       " innit ": " is it not ",
       " i've ": " i have ",
       " isn't ": " is not ",
       " isnt ": " is not ",
       " it'd ": " it would ",
       " it'll ": " it will ",
       " it's ": " it is ",
       " let's ": " let us ",
       " ma'am ": " madam ",
       " mayn't ": " may not ",
       " may've ": " may have ",
       " methinks": " me thinks ",
       " mightn't ": " might not ",
       " might've ": " might have ",
       " mustn't ": " must not ",
       " mustn't've ": " must not have ",
       " must've ": " must have ",
       " needn't ": " need not ",
       " ne'er ": " never ",
       " o'clock ": " of the clock ",
       " o'er ": " over ",
       " ol' ": " old ",
       " oughtn't ": " ought not ",
       "'s ": " is ",
       " shalln't ": " shall not ",
       " shan't ": " shall not ",
       " she'd ": " she would ",
       " she'll ": " she shall ",
       " she'll ": " she will ",
       " she's ": " she has ",
       " she's ": " she is ",
       " should've ": " should have ",
       " shouldn't ": " should not ",
       " shouldn't've ": " should not have ",
       " somebody's ": " somebody has ",
       " somebody's ": " somebody is ",
       " someone's ": " someone has ",
       " someone's ": " someone is ",
       " something's ": " something has ",
       " something's ": " something is ",
       " so're ": " so are ",
       " that'll ": " that shall ",
       " that'll ": " that will ",
       " that're ": " that are ",
       " tht's ": " that is ",
       " tht's ": " that has",
       " that's ": " that has ",
       " that's ": " that is ",
       " that'd ": " that would ",
       " that'd ": " that had ",
       " there'd ": " there had ",
       " there'd ": " there would ",
       " there'll ": " there shall ",
       " there'll ": " there will ",
       "there're ": " there are ",
       " there's ": " there has ",
       " there's ": " there is ",
       " these're": " these are ",
       " these've": " these have ",
       " they'd ": " they had ",
       " they'd ": " they would ",
       " they'll ": " they shall ",
       " they'll ": " they will ",
       " they're ": " they are ",
       " they're ": " they were ",
       " they've ": " they have ",
       " this's ": " this has ",
       " this's ": " this is ",
       " those're ": " those are ",
       " those've ": " those have ",
       " tho ": " though ",
       " 'tis ": " it is ",
       " to've ": " to have ",
       " 'twas ": " it was ",
       " wanna ": " want to ",
       " wasn't ": " was not ",
       " we'd ": " we had ",
       " we'd ": " we would ",
       " we'd ": " we did ",
       " we'll ": " we shall ",
       " we'll ": " we will ",
       " we're ": " we are ",
       " we've ": " we have ",
       " weren't ": " were not ",
       " what'd ": " what did ",
       " what'll ": " what shall ",
       " what'll ": " what will ",
       " what're ": " what are ",
       " what're ": " what were ",
       " what's ": " what has ",
       " what's ": " what is ",
       " what's ": " what does ",
       " what've ": " what have ",
       " when's ": " when has ",
       " when's ": " when is ",
       " where'd ": " where did ",
       " where'll ": " where shall ",
       " where'll ": " where will ",
       " where're ": " where are ",
       " where's ": " where has ",
       " where's ": " where is ",
       " where's ": " where does ",
       " where've ": " where have ",
       " which'd ": " which had ",
       " which'd ": " which would ",
       " which'll ": " which shall ",
       " which'll ": " which will ",
       " which're ": " which are ",
       " which's ": " which has ",
       " which's ": " which is ",
       " which've ": " which have ",
       " who'd ": " who would ",
       " who'd ": " who had ",
       " who'd ": " who did ",
       " who'd've ": " who would have ",
       " who'll ": " who shall ",
       " who'll ": " who will ",
       " who're ": " who are ",
       " who's ": " who has ",
       " who's ": " who is ",
       " who's ": " who does ",
       " who've ": " who have ",
       " why'd ": " why did ",
       " why're ": " why are ",
       " why's ": " why has ",
       " why's ": " why is ",
       " why's ": " why does ",
       " wit' ": " with ",
       " won't ": " will not ",
       " would've ": " would have ",
       " wouldn't ": " would not ",
       " wouldn't've ": " would not have ",
       " y'all ": " you all ",
       " y'all'd've ": " you all would have ",
       " y'all'dn't've'd ": " you all would not have had ",
       " y'all're ": " you all are ",
       " you'd ": " you had ",
       " you'd ": " you would ",
       " you'll ": " you shall ",
       " you'll ": " you will ",
       " you're ": " you are ",
       "you're ": " you are ",
       " you've ": " you have ",
       " u ": " you ",
       " ur ": " your ",
       " n ": " and ",
       " wbu ": " what about you ",
       " omg ": " oh my god ",
       " kno ": " know ",
       " d ": " the ",
       " r ": " are ",
       " miss'n ": " missing ",
       " missin ": " missing ",
       " fml ": " fuck my life ",
       " fam ": " family ",
       " thaanks ": " thank you ",
       " dinenr ": " dinner ",
       " wbuu": " what about you",
       " yawwwnn ": " yawn ",
       " sooo ": " so ",
       " whyyyyyyyy ": "why ",
       " tm ": " trust me",
       "tm ": " trust me",
       " doa ": " dead on arrival ",
       " callin ": " calling "
   }  # copied from https://www.kaggle.com/raymant/text-processing-sentiment-analysis?scriptVersionId=33503187&cellId=23.

   # Added some additional abbreviations for this dataset

   def cont_to_exp(stringtopre):
       if type(stringtopre) is str:
           for key in contractions:
               value = contractions[key]
               stringtopre = stringtopre.replace(key, value)
           return stringtopre
       else:
           return stringtopre

   stringtopre = cont_to_exp(stringtopre)  # Fix abbreviations
   stringtopre = " ".join([t for t in stringtopre.split() if t not in STOP_WORDS])  # Remove stop words. See https://github.com/explosion/spaCy/blob/master/spacy/lang/en/stop_words.py to see the full list
   stringtopre = re.sub('[^a-zA-Z]', " ", stringtopre)  # Remove non-alphabetical characters
   stringtopre = re.sub(r'\s+', ' ', stringtopre)  # Remove extra space between words
   stringtopre = " ".join([w for w in stringtopre.split() if len(w) > 3])  # Removing short words that are 3 characters or less
   return stringtopre

def predictText(text):
   x_test = preprocess_string(text)
   ayo = cv.transform([x_test])
   if model.predict(ayo) == 1:
       result ='The string "' + text + '" is positive!'
   elif model.predict(ayo) == 0:
       result = 'The string "' + text + '" is negative!'
   return result


class responseWidget(QFrame, Ui_ResponseWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class FirstApp(Ui_MainWindow):
    def __init__(self, window):
        self.setupUi(window)
        self.addResponse('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n<html><head><meta name="qrichtext" content="1" /><style type="text/css">\np, li { white-space: pre-wrap; }\n</style></head><body style=" font-family:\'Segoe UI\'; font-size:20px; font-weight:400; font-style:normal;">\n<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:20px; color:#ffffff;">Shiekh Islam</span><span style=" font-size:20px;"> </span><img src="C:/Users/Rachel Liang/Documents/pyQtExperimentation/Resources/verifiedtwitter.ico" /><span style=" font-size:20px;"> </span><span style=" font-size:20px; color:#8899a6;">@SsHIs </span><span style=" font-size:20px;"> </span></p>\n<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:20px; color:#ffffff;">' \
                + 'Try out the Twitter Sentiment Analysis by adding some text above and pressing the tweet button!' + '</span></p></body></html>')
        self.tweetButton.setEnabled(True)
        self.tweetButton.clicked.connect(lambda:self.buttonClick())
    #     self.textEdit.keyPressEvent(QKeyEvent)
    #     self.textEdit.document().isEmpty()
    #
    # def keyPressEvent(self, event):
    #     self.checkstatus()
    #     print('gothere')
    #
    # def checkstatus(self):
    #     if self.textEdit.text() == "":
    #         self.tweetButton.setEnabled(False)
    #     else:
    #         self.tweetButton.setEnabled(True)

    def buttonClick(self):
        inputedText = self.textEdit.toPlainText()
        if inputedText != "":
            self.showText(predictText(inputedText))
            self.textEdit.clear()

    def showText(self, _str):
        text = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n<html><head><meta name="qrichtext" content="1" /><style type="text/css">\np, li { white-space: pre-wrap; }\n</style></head><body style=" font-family:\'Segoe UI\'; font-size:20px; font-weight:400; font-style:normal;">\n<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:20px; color:#ffffff;">Shiekh Islam</span><span style=" font-size:20px;"> </span><img src="C:/Users/Rachel Liang/Documents/pyQtExperimentation/Resources/verifiedtwitter.ico" /><span style=" font-size:20px;"> </span><span style=" font-size:20px; color:#8899a6;">@SsHIs </span><span style=" font-size:20px;"> </span></p>\n<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-size:20px; color:#ffffff;">' \
               + _str + '</span></p></body></html>'
        previousText = text
        self.addResponse(previousText)

    def addResponse(self, text):
        Item = QtWidgets.QListWidgetItem()
        Item_Widget = responseWidget()
        Item_Widget.textBrowser.setHtml(text)
        Item.setSizeHint(Item_Widget.size())
        self.listWidget.insertItem(0, Item)
        self.listWidget.setItemWidget(Item, Item_Widget)


app = QtWidgets.QApplication(sys.argv)
#form = QtWidgets.QWidget()
MainWindow = QtWidgets.QMainWindow()

# Create an instance of our app!
#uiForm = responseWidget(form)
ui = FirstApp(MainWindow)

#show the window and start the app
MainWindow.show()
app.exec_()