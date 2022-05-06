from __future__ import print_function

import pickle
import re
import warnings
import os
import torch

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from spacy.lang.en import STOP_WORDS

warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('wordnet')

col_name = ['Polarity', 'Tweet']  # Label all the columns
tweet = pd.read_csv(r"Untitled spreadsheet - Sheet1.csv",
                    encoding='latin', header=None, names=col_name)  # Read csv
tweet = tweet.sample(frac=1)  # Sample all rows. frac=1 means return all rows in random order
size = 328*2
tweet = tweet[
        :size]  # out of 1.6 million it takes a sample.

tweet['Polarity'] = tweet['Polarity'].replace(4,1)  # Since neutral will not be tested, for simplicity it will be classified: 0 as negative and 1 as positive

tweet['Tweet'] = tweet['Tweet'].astype('str')  # converting pandas object to a string type

tweet.Tweet = tweet.Tweet.apply(lambda x: re.sub(r'http?:\/\/\S+', '', x))  # Remove links with https
tweet.Tweet.apply(lambda x: re.sub(r"www\.[a-z]?\.?@[\w]+(com)+|[a-z]+\.(com)", '', x))  # Remove links with www. and com

tweet['Tweet'] = tweet.Tweet.str.lower()  # Lower case all tweets
tweet['Tweet'] = tweet.Tweet.apply(lambda x: re.sub('@[^\s]+', '', x))  # Remove @username
tweet['Tweet'] = tweet.Tweet.apply(lambda x: re.sub("#[A-Za-z0-9_]", '', x))  # Remove #hashtag
tweet['Tweet'] = tweet['Tweet'].apply(lambda x: re.sub(' RT ', "", x))  # Remove RT (Retweet)

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
    " callin ": " calling ",
    "omgee": "oh my god",

}  # copied from https://www.kaggle.com/raymant/text-processing-sentiment-analysis?scriptVersionId=33503187&cellId=23.


# Added some additional abbreviations for this dataset

def cont_to_exp(tweet):
    if type(tweet) is str:
        for key in contractions:
            value = contractions[key]
            tweet = tweet.replace(key, value)
        return tweet
    else:
        return tweet


tweet['Tweet'] = tweet['Tweet'].apply(lambda x: cont_to_exp(x))  # Fix abbreviations
tweet['Tweet'] = tweet['Tweet'].apply(lambda x: " ".join([t for t in x.split() if
                                                          t not in STOP_WORDS]))  # Remove stop words. See https://github.com/explosion/spaCy/blob/master/spacy/lang/en/stop_words.py to see the full list
tweet['Tweet'] = tweet.Tweet.apply(lambda x: re.sub('[^a-zA-Z]', " ", x))  # Remove non-alphabetical characters
tweet['Tweet'] = tweet['Tweet'].apply(lambda x: " ".join(x.split()))  # Remove extra space between words
tweet['Tweet'] = tweet['Tweet'].apply(
    lambda x: " ".join([w for w in x.split() if len(w) > 3]))  # Removing short words that are 3 characters or less

# csv = tweet.to_csv('Preproccessed2.csv', index = False)

gross = tweet['Tweet']
tweet = shuffle(tweet).reset_index(drop=True)  # Reset the index after shuffling

# Splitting the dataset
x = tweet['Tweet']  # changed from tweet['Tweet'], needed for CNB.fit to work!
y = tweet['Polarity']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=19)

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(strip_accents='ascii', stop_words='english')
X_train_tf = tf.fit_transform(x_train)

# transform the test set with vectoriser
X_test_tf = tf.transform(x_test)
x_tf = tf.transform(x)

# Complement Naive Baiyes
CNB = ComplementNB()
CNB.fit(X_train_tf, y_train)
CNB_cross_validation = cross_val_score(CNB, x_tf, y, n_jobs=-1)
print("CNB Cross Validation score = ", CNB_cross_validation)
print("CNB Train accuracy = {:.2f}%".format(CNB.score(X_train_tf, y_train) * 100))
print("CNB Test accuracy = {:.2f}%".format(CNB.score(X_test_tf, y_test) * 100))
CNB_train = CNB.score(X_train_tf, y_train)
CNB_test = CNB.score(X_test_tf, y_test)
print("CNB Confusion Matrix: ")
CNB_complete = [CNB_train, CNB_test]
print(" ")

predict_CNB = CNB.predict(X_test_tf)  # Predict test

print(confusion_matrix(y_test, predict_CNB))  # Print confusion matrix
print(classification_report(y_test, predict_CNB))  # Performance check using Complement Naive Bayes


# Multinomial Naive Bayes
MNB = MultinomialNB()
MNB.fit(X_train_tf, y_train)
MNB_cross_validation = cross_val_score(MNB, x_tf, y, n_jobs=-1)
print("MNB Cross Validation score = ", MNB_cross_validation)
print("MNB Train accuracy = {:.2f}%".format(MNB.score(X_train_tf, y_train) * 100))
print("MNB Test accuracy = {:.2f}%".format(MNB.score(X_test_tf, y_test) * 100))
MNB_train = MNB.score(X_train_tf, y_train)
MNB_test = MNB.score(X_test_tf, y_test)
print("MNB Confusion Matrix: ")
MNB_complete = [MNB_train, MNB_test]
print(" ")

predict_MNB = MNB.predict(X_test_tf)  # Predict test

print(confusion_matrix(y_test, predict_MNB))  # Print confusion matrix
print(classification_report(y_test, predict_MNB))  # Performance check using Multinomial Naive Bayes

# Bernoulli Naive Bayes
BNB = BernoulliNB()
BNB.fit(X_train_tf, y_train)
BNB_cross_validation = cross_val_score(BNB, x_tf, y, n_jobs=-1)
print("BNB Cross Validation score = ", BNB_cross_validation)
print("BNB Train accuracy = {:.2f}%".format(BNB.score(X_train_tf, y_train) * 100))
print("BNB Test accuracy = {:.2f}%".format(BNB.score(X_test_tf, y_test) * 100))
BNB_train = BNB.score(X_train_tf, y_train)
BNB_test = BNB.score(X_test_tf, y_test)
print("BNB Confusion Matrix: ")
BNB_complete = [BNB_train, BNB_test]
print(" ")

predict_BNB = BNB.predict(X_test_tf)  # Predict test

print(confusion_matrix(y_test, predict_BNB))  # Print confusion matrix
print(classification_report(y_test, predict_BNB))  # Performance check using Multinomial Naive Bayes

# Make a bar graph
label = ['Train Accuracy', 'Test Accuracy']
plt.xticks(range(len(CNB_complete)), label)
plt.ylabel('Accuracy')
plt.title('CNB Accuracy bar graph for a sample of ' + str(size))
plt.bar(range(len(CNB_complete)), CNB_complete, color=['pink', 'black'])
Train_acc = mpatches.Patch(color='pink', label='Train Accuracy')
Test_acc = mpatches.Patch(color='black', label='Test Accuracy')
plt.legend(handles=[Train_acc, Test_acc], loc='best')
plt.gcf().set_size_inches(10, 10)
if not os.path.isdir('outputs/accuracy'):
    os.makedirs('outputs/accuracy')
plt.savefig('outputs/accuracy/train_and_test_accuracy_CNB')
plt.clf()

label = ['Train Accuracy', 'Test Accuracy']
plt.xticks(range(len(MNB_complete)), label)
plt.ylabel('Accuracy')
plt.title('MNB Accuracy bar graph for a sample of ' + str(size))
plt.bar(range(len(MNB_complete)), MNB_complete, color=['pink', 'black'])
Train_acc = mpatches.Patch(color='pink', label='Train Accuracy')
Test_acc = mpatches.Patch(color='black', label='Test Accuracy')
plt.legend(handles=[Train_acc, Test_acc], loc='best')
plt.gcf().set_size_inches(10, 10)
plt.savefig('outputs/accuracy/train_and_test_accuracy_MNB')
plt.clf()

label = ['Train Accuracy', 'Test Accuracy']
plt.xticks(range(len(BNB_complete)), label)
plt.ylabel('Accuracy')
plt.title('BNB Accuracy bar graph for a sample of ' + str(size))
plt.bar(range(len(BNB_complete)), BNB_complete, color=['pink', 'black'])
Train_acc = mpatches.Patch(color='pink', label='Train Accuracy')
Test_acc = mpatches.Patch(color='black', label='Test Accuracy')
plt.legend(handles=[Train_acc, Test_acc], loc='best')
plt.gcf().set_size_inches(10, 10)
plt.savefig('outputs/accuracy/train_and_test_accuracy_BNB')
plt.clf()

if not os.path.isdir('outputs/pickled'):
    os.makedirs('outputs/pickled')
with open('outputs/pickled/CNB_model', 'wb') as f:
    pickle.dump(CNB, f)
with open('outputs/pickled/MNB_model', 'wb') as f:
    pickle.dump(MNB, f)
with open('outputs/pickled/BNB_model', 'wb') as f:
    pickle.dump(BNB, f)
with open('outputs/pickled/cv', 'wb') as f:
    pickle.dump(tf, f)

from sklearn.metrics import roc_curve

if not os.path.isdir('outputs/ROC'):
    os.makedirs('outputs/ROC')

fpr_dt_1, tpr_dt_1, _ = roc_curve(y_test, CNB.predict_proba(X_test_tf)[:, 1])
plt.plot(fpr_dt_1, tpr_dt_1, label="ROC curve CNB")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.gcf().set_size_inches(8, 8)
plt.savefig('outputs/ROC/ROC_curve_CNB')
plt.clf()

fpr_dt_2, tpr_dt_2, _ = roc_curve(y_test, MNB.predict_proba(X_test_tf)[:, 1])
plt.plot(fpr_dt_2, tpr_dt_2, label="ROC curve MNB")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.gcf().set_size_inches(8, 8)
plt.savefig('outputs/ROC/ROC_curve_MNB')
plt.clf()

fpr_dt_3, tpr_dt_3, _ = roc_curve(y_test, BNB.predict_proba(X_test_tf)[:, 1])
plt.plot(fpr_dt_3, tpr_dt_3, label="ROC curve BNB")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.gcf().set_size_inches(8, 8)
plt.savefig('outputs/ROC/ROC_curve_BNB')
plt.clf()

ROC_score_CNB = roc_auc_score(y_test, predict_CNB)  # Checking performance using ROC Score
print("CNB Area Under the Curve = ", ROC_score_CNB)
print(" ")

ROC_score_MNB = roc_auc_score(y_test, predict_MNB)  # Checking performance using ROC Score
print("MNB Area Under the Curve = ", ROC_score_MNB)
print(" ")

ROC_score_BNB = roc_auc_score(y_test, predict_BNB)  # Checking performance using ROC Score
print("BNB Area Under the Curve = ", ROC_score_BNB)
print(" ")

from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

# converting the vectoriser and model into a pipeline
# this is necessary as LIME takes a model pipeline as an input
c = make_pipeline(tf, CNB)
d = make_pipeline(tf, MNB)
e = make_pipeline(tf, BNB)
# saving a list of strings version of the X_test object
# print(gross)
ls_X_test = list(x_test)

# print(ls_X_test)
# saving the class names in a dictionary to increase interpretability
class_names = {0:'negative', 1:'positive'}

LIME_explainer = LimeTextExplainer(class_names=class_names)

# choose a random single prediction
idx = 0
print(ls_X_test[idx])
# explain the chosen prediction
# use the probability results of the logistic regression
# can also add num_features parameter to reduce the number of features explained
LIME_exp_CNB = LIME_explainer.explain_instance(ls_X_test[idx], c.predict_proba)
LIME_exp_MNB = LIME_explainer.explain_instance(ls_X_test[idx], d.predict_proba)
LIME_exp_BNB = LIME_explainer.explain_instance(ls_X_test[idx], e.predict_proba)


# print results
print('Document id: %d' % idx)
print('Tweet: ', ls_X_test[idx])
print('Positivity =', c.predict_proba([ls_X_test[idx]]).round(3)[0, 1])
print('True class: %s' % class_names.get(list(y_test)[idx]))
print(" ")

print('Document id: %d' % idx)
print('Tweet: ', ls_X_test[idx])
print('Positivity =', d.predict_proba([ls_X_test[idx]]).round(3)[0, 1])
print('True class: %s' % class_names.get(list(y_test)[idx]))
print(" ")

print('Document id: %d' % idx)
print('Tweet: ', ls_X_test[idx])
print('Positivity =', e.predict_proba([ls_X_test[idx]]).round(3)[0, 1])
print('True class: %s' % class_names.get(list(y_test)[idx]))
print(" ")

if not os.path.isdir('outputs/lime'):
    os.makedirs('outputs/lime')

LIME_exp_CNB.save_to_file('outputs/lime/lime_CNB.html')
LIME_exp_MNB.save_to_file('outputs/lime/lime_MNB.html')
LIME_exp_BNB.save_to_file('outputs/lime/lime_BNB.html')

LIME_exp_CNB.as_pyplot_figure()
plt.savefig('outputs/lime/lime_CNB_bargraph')
plt.clf()

LIME_exp_MNB.as_pyplot_figure()
plt.savefig('outputs/lime/lime_MNB_bargraph')
plt.clf()

LIME_exp_BNB.as_pyplot_figure()
plt.savefig('outputs/lime/lime_BNB_bargraph')
plt.clf()

from wordcloud import WordCloud

tweet_pos = tweet[tweet['Polarity'] == 1] # Only collect tweets that are positive
tweet_neg = tweet[tweet['Polarity'] == 0] # Only collect tweets that are negative

if not os.path.isdir('outputs/wordcloud'):
    os.makedirs('outputs/wordcloud')

plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(tweet_neg['Tweet']))
plt.imshow(wc , interpolation = 'bilinear')
plt.savefig("outputs/wordcloud/word_cloud_negative")

plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(tweet_pos['Tweet']))
plt.imshow(wc , interpolation = 'bilinear')
plt.savefig("outputs/wordcloud/word_cloud_positive")

#img processing confidence
img_path = 'img.jpg'

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
results = model(img_path)

txt_file2 = open('outputs/output.txt','w')
content2 = txt_file2.write(str(results.pandas().xyxy[0]))
txt_file2.close();

threats = ['knife', 'gun', 'bullet', 'knives', 'blood', 'knifes', 'guns', 'bullets', 'bloods']

#if os.path.isdir('outputs/output.txt'):
lines = []
conf = []
avg = 0
counter =0
with open('outputs/output.txt') as f:
    lines = f.readlines()
if (len(lines) >1):
    for i in range (1, len(lines)):
        sp = lines[i].split()
        print(sp[7])
        print(sp)
        print(str(sp[5]))
        if any(ext in sp[7] for ext in threats):
            avg += float(sp[5])
            counter += 1
if counter>0:
    average = avg/counter
    if (average > 0.5):
        text_file = open("outputs/output.txt", "a")
        text_file.write("\nthreat found: " + str(average))
        text_file.close()
    else:
        text_file = open("outputs/output.txt", "a")
        text_file.write("\nthreat found but not confident enough: " + str(average))
        text_file.close()
else:
    text_file = open("outputs/output.txt", "a")
    text_file.write("\nno threats found")
    text_file.close()

multimomdalNum = (1-(c.predict_proba([ls_X_test[idx]]).round(3)[0, 1]) + average)/2
if (multimomdalNum > 0.5):
    print("Warning, threat was found from this social media post. Confidence: " + str(multimomdalNum))
else:
    print("This social media post is normal. Confidence: " + str(multimomdalNum))

print("DONE!")
