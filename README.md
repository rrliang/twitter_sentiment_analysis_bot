# twitter_sentiment_analysis_bot
## For the University of Hartford, CS 368 Data Mining class.  
### Created by Alana Cedeno and Rachel Liang.
-----------------------------------------
This repository contains trained models on Twitter sentiment analysis.
It has a GUI to utilize the trained data models.  

## Preprocess and train
To preprocess and train the model using NaiveBayes, run the preprocessingandtraining.py file.
This should output a cv, model_pickle, ROC Curve.png, and Train and test accuracy.png.  

## Testing and GUI
To run GUI on the trained data, run the firstApp.py. This file uses the Gui.ui, and responsewidget.ui files, which can be opened and edited on Qt Designer.  
Additionally, the firstApp.py requires the ui files to be transformed into py files, so in a terminal where PyQT5 has been installed, run the command:  
pyuic5 filename.ui -o filename.py  
Whenever a change is made to the ui files.  
To make a qrc to a py file,  
pyrcc5 twitter.qrc -o twitter.py  

WARNING- in order to run everything correctly, please clone the repository as the CSV and wkhtmltoimage.exe do not correctly download if you just download the repo as zip. If you do not want to clone the repository, download as zip, and replace the CSV from this Kaggle link: https://www.kaggle.com/kazanova/sentiment140 and you can download the wkhtmltoimage.exe from: https://drive.google.com/file/d/1vXJI05UujZq7ER2H-d_fJwkdEeciOSMh/view?usp=sharing

## Packages to install beforehand
Additionally, included packages that must be installed:  
- pickle  
- re  
- warnings  
- matplotlib  
- nltk  
- pandas  
- sklearn  
- spacy  
- lime  
- wordcloud  
- PyQt5  
- sys  
- imgkit  

## Related files to directory and processes
```bash
twitter_sentiment_analysis_bot
├───Preprocess and train
│   └───preprocessingandtraining.py
│       ├───Inputs:
│       │   └───training.1600000.processed.noemoticon.csv
│       └───Outputs:
│           ├───ROC
│           │   ├───ROC_curve_BNB.png
│           │   ├───ROC_curve_CNB.png
│           │   └───ROC_curve_MNB.png
│           ├───wordcloud
│           │   ├───word_cloud_negative.png
│           │   └───word_cloud_positive.png
│           ├───accuracy
│           │   ├───train_and_test_accuracy_BNB.png
│           │   ├───train_and_test_accuracy_CNB.png
│           │   └───train_and_test_accuracy_MNB.png
│           ├───lime
│           │   ├───lime_BNB.png
│           │   ├───lime_BNB_bargraph.png
│           │   ├───lime_CNB.png
│           │   ├───lime_CNB_bargraph.png
│           │   ├───lime_BNB.png
│           │   └───lime_BNB_bargraph.png
│           └───pickled
│               ├───BNB_model
│               ├───CNB_model
│               ├───MNB_model
│               └───cv
├───QT Designer (GUI Building using PyQT5)
│   ├───Inputs:
│   │    └───Resources
│   │        ├───drislampfp.ico
│   │        ├───twitterpfp.ico
│   │        └───verifiedtwitter.ico
│   └───Outputs:
│       ├───twitter.qrc
│       ├───Gui.ui
│       └───responsewidget.ui
└───Testing and GUI
    └───firstApp.py
        ├───Inputs:
        │   ├───cv
        │   ├───model_pickle
        │   ├───gui.py
        │   ├───twitter_rc.py
        │   ├───wkhtmltoimage.exe        
        │   └───responsewidget.py
        └───Outputs:
            ├───GUI
            └───Resources
                ├───lime.html
                └───lime.png
```
