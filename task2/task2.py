import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset):
    dataset = dataset.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
    dataset.Age = dataset.Age.astype(int)
    dataset["Fare"]= dataset["Fare"].fillna(dataset["Fare"].median())
    dataset.Sex = dataset.Sex.replace({"female":0, "male":1})
    return dataset

train_ds = prepare_dataset(pd.read_csv("data/train.csv"))

X = train_ds.iloc[:,2:9]
y = train_ds.Survived.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

np.random.seed(42)

kneighbors_clf = KNeighborsClassifier().fit(X_train,y_train)
kneighbors_pred = kneighbors_clf.predict(X_test)
kneighbors_rprt = classification_report(y_test,kneighbors_pred)

kneighbors_score = kneighbors_clf.score(X_test,y_test)
kneighbors_cm = confusion_matrix(list(y_test),kneighbors_pred)

print("\nKNeighbors Classifier:")
print("\nConfusion Matrix: \n{} \nScore: {}".format(kneighbors_cm,kneighbors_score))

decision_tree_clf = DecisionTreeClassifier().fit(X_train,y_train)
decision_tree_pred = decision_tree_clf.predict(X_test)
decision_tree_rprt = classification_report(y_test,decision_tree_pred)

decision_tree_score = decision_tree_clf.score(X_test,y_test)
decision_tree_cm = confusion_matrix(list(y_test),decision_tree_pred)

print("\nDecision Tree Classifier:")
print("\nConfusion Matrix: \n{} \nScore: {}".format(decision_tree_cm,decision_tree_score))


logisticreg_clf = LogisticRegression().fit(X_train,y_train)
logisticreg_pred = logisticreg_clf.predict(X_test)
logisticreg_rprt = classification_report(y_test,logisticreg_pred)

logisticreg_score = logisticreg_clf.score(X_test,y_test)
logisticreg_cm = confusion_matrix(list(y_test),logisticreg_pred)

print("\Logistic Regression Classifier:")
print("\nConfusion Matrix: \n{} \nScore: {}".format(logisticreg_cm,logisticreg_score))

test_ds = prepare_dataset(pd.read_csv("data/test.csv"))

X = test_ds.iloc[:,1:7]

decision_tree_test = decision_tree_clf.predict(X)
# print(decision_tree_test)