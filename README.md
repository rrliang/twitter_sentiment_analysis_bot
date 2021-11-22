# twitter_sentiment_analysis_bot
For Data Mining.
Created by Alana Cedeno and Rachel Liang.
-----------------------------------------
This repository contains trained models on Twitter sentiment analysis.
It has a GUI to utilize the trained data models.

To preprocess and train the model using NaiveBayes, run the preprocessingandtraining.py file.
This should output a cv, model_pickle, ROC Curve.png, and Train and test accuracy.png.

To run GUI on the trained data, run the firstApp.py. This file uses the Gui.ui, and responsewidget.ui files, which can be opened and edited on Qt Designer.
Additionally, the firstApp.py requires the ui files to be transformed into py files, so in a terminal where PyQT5 has been installed, run the command:
pyuic5 filename.ui -o filename.py
Whenever a change is made to the ui files.
To make a qrc to a py file,
pyrcc5 twitter.qrc -o twitter.py

```bash
twitter_sentiment_analysis_bot
├───Preprocess and train
│   └───preprocessingandtraining.py
│       ├───Inputs:
│       │   └───training.1600000.processed.noemoticon.csv
│       └───Outputs:
│           ├───cv
│           ├───model_pickle
│           ├───ROC Curve.png
│           ├───lime.html
│           ├───Lime Bargraph.png
│           └───Train and test accuracy.png
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
        │   └───responsewidget.py
        └───Outputs:
            └───GUI
```
