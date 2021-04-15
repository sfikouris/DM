import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sklearn as sk
from sklearn import datasets
from sklearn import svm
import pandas as pd


df = pd.read_csv('heart.csv', sep= ',', header =0)

#cleaning dataset
df.loc[df['famhist']=='Present','famhist'] = 1
df.loc[df['famhist']=='Absent','famhist'] = 0

#create data 
y = df['chd']
X = df.drop('chd',axis=1)

#preparing data for train
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#apply svm model with cross-valitation and train the algorithm
clf = svm.SVC(kernel='linear',C=1).fit(X_train, y_train)
clf_with_CV = svm.SVC(kernel='linear',C=1,random_state=42).fit(X_train, y_train)

#making prediction
y_pred = clf_with_CV.predict(X_test)
#print(y_pred)

#evaluating the Algorithm
print("SVM\n")
scores = cross_val_score(clf_with_CV, X_test,y_test,cv=5)
print("Confusion Matrix\n {}\n".format(confusion_matrix(y_test,y_pred)))
print(classification_report(y_test,y_pred))
print("Accuracy score: {}".format(accuracy_score(y_test, y_pred)))


#print(scores)
print("%0.2f score mean with a standard deviation of %0.2f" %(scores.mean(), scores.std()))
#print(clf.score(X_train, y_train))

#print("PRININT Y")
#print(y)
#print("prinintg X")
#print(X.to_string())

#Random Forest for Regression

#training the algorithm
regressor = RandomForestClassifier(max_depth=2, random_state=42).fit(X_train,y_train)
y_pred_forest = regressor.predict(X_test)
scores_forest = cross_val_score(regressor, X_test,y_test,cv=5)


#print(scores_forest)

print("\nRandom Forest\n")
print("Confusion Matrix\n {}\n".format(confusion_matrix(y_test,y_pred_forest)))
print(classification_report(y_test,y_pred_forest))
print("Accuracy score: {}".format(accuracy_score(y_test, y_pred_forest)))
print("%0.2f score mean with a standard deviation of %0.2f" %(scores_forest.mean(), scores_forest.std()))



