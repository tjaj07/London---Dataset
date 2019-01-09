import numpy as np
import pandas as pd

# Importing the dataset 
column = list(np.arange(40))
X = pd.read_csv('train.csv',names = column,header = None)
Y = pd.read_csv('trainLabels.csv',names = ["Temp"],header = None)
Y = Y['Temp']

X.info()

# Spliting the data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size = 0.8,random_state = 42)

# Predict
# Fitting the Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

# checking the predictions
print(round(classifier.score(x_test,y_test) * 100, 2))
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
print(confusion_matrix(y_test,classifier.predict(x_test)))
print(accuracy_score(y_test, classifier.predict(x_test)))
print(classification_report(y_test, classifier.predict(x_test)))

from sklearn.model_selection import cross_val_score
res = cross_val_score(classifier,x_test, y_test, cv=5, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
