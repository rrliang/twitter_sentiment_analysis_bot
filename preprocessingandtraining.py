import nltk
import warnings
import re
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from spacy.lang.en import STOP_WORDS
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import *

warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('wordnet')

col_name = ['Polarity', 'ID', 'Time', 'Query', 'Username', 'Tweet']  # Label all the columns
tweet = pd.read_csv(r"training.1600000.processed.noemoticon.csv",
                  encoding='latin', header=None, names=col_name)  # Read csv
tweet = tweet.sample(frac=1)  # Sample all rows. frac=1 means return all rows in random order
tweet = tweet[
      :200000]  # out of 1.6 million it takes a sample. Tried 1.6 million, 800,000 , and 200,000 to compare ROC curve and CNB acurracy.
tweet['Polarity'] = tweet['Polarity'].replace(4,
                                            1)  # Since neutral will not be tested, for simplicity it will be classified: 0 as negative and 1 as positive

tweet['Tweet'] = tweet['Tweet'].astype('str')  # converting pandas object to a string type

tweet.Tweet = tweet.Tweet.apply(lambda x: re.sub(r'http?:\/\/\S+', '', x))  # Remove links with https
tweet.Tweet.apply(
  lambda x: re.sub(r"www\.[a-z]?\.?@[\w]+(com)+|[a-z]+\.(com)", '', x))  # Remove links with www. and com

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
  " callin ": " calling "

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

tweet = shuffle(tweet).reset_index(drop=True)  # Reset the index after shuffling

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)  # Tokenize tweet
tweet_counts = cv.fit_transform(tweet['Tweet'].values.astype('U'))  # Convert the data to Unicode

# Splitting the dataset
x = tweet_counts  # changed from tweet['Tweet'], needed for CNB.fit to work!
y = tweet['Polarity']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=19)

# Complement Naive Baiyes
CNB = ComplementNB()
CNB.fit(x_train, y_train)
CNB_cross_validation = cross_val_score(CNB, x, y, n_jobs=-1)
print("Cross Validation score = ", CNB_cross_validation)
print("Train accuracy = {:.2f}%".format(CNB.score(x_train, y_train) * 100))
print("Test accuracy = {:.2f}%".format(CNB.score(x_test, y_test) * 100))
CNB_train = CNB.score(x_train, y_train)
CNB_test = CNB.score(x_test, y_test)
CNB_complete = [CNB_train, CNB_test]

# Make a bar graph
label = ['Train Accuracy', 'Test Accuracy']
plt.xticks(range(len(CNB_complete)), label)
plt.ylabel('Accuracy')
plt.title('Accuracy bar graph for a sample of 200,000')
plt.bar(range(len(CNB_complete)), CNB_complete, color=['pink', 'black'])
Train_acc = mpatches.Patch(color='pink', label='Train Accuracy')
Test_acc = mpatches.Patch(color='black', label='Test Accuracy')
plt.legend(handles=[Train_acc, Test_acc], loc='best')
plt.gcf().set_size_inches(10, 10)
plt.savefig('Train and test accuracy')


predict_CNB = CNB.predict(x_test)  # Predict test

with open('model_pickle', 'wb') as f:
  pickle.dump(CNB, f)
with open('cv', 'wb') as f:
  pickle.dump(cv, f)

print(confusion_matrix(y_test, predict_CNB))  # Print confusion matrix
print(classification_report(y_test, predict_CNB)) # Performance check using Complement Naive Bayes

plt.clf()
from sklearn.metrics import roc_curve
fpr_dt_1, tpr_dt_1,_=roc_curve(y_test,CNB.predict_proba(x_test)[:,1])
plt.plot(fpr_dt_1,tpr_dt_1,label="ROC curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.gcf().set_size_inches(8, 8)
plt.savefig('ROC Curve')

ROC_score = roc_auc_score(y_test, predict_CNB) # Checking performance using ROC Score
print("Area Under the Curve = ", ROC_score)
