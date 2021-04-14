import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import sklearn as sk
from sklearn import datasets
from sklearn import svm
import pandas as pd


df = pd.read_csv('heart.csv', sep= ',', header =0)


df.loc[df['famhist']=='Present','famhist'] = 1
df.loc[df['famhist']=='Absent','famhist'] = 0

#print(df.head())
#y = df.iloc[:,9]
y = df['chd']
#X = df.iloc[:,:9]
X = df.drop('chd',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

clf = svm.SVC(kernel='linear',C=1).fit(X_train, y_train)
clf_with_CV = svm.SVC(kernel='linear',C=1,random_state=40).fit(X_train, y_train)
scores = cross_val_score(clf_with_CV, X_test,y_test,cv=5)

#making prediction
y_pred = clf_with_CV.predict(X_test)
#print(y_pred)

#evaluating the Algorithm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" %(scores.mean(), scores.std()))
#print(clf.score(X_train, y_train))


#print("PRININT Y")
#print(y)
#print("prinintg X")
#print(X.to_string())




#SVM = svm.LinearSVC()
#print(SVM)
#SVM.fit(X,y)
#SVM.predict(X.iloc[460:,:])
#round(SVM.score(X,y),4)