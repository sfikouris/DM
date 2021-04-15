import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset):
    dataset = dataset.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
    dataset.Sex = dataset.Sex.replace({"female":0, "male":1})
    return dataset

train_ds = prepare_dataset(pd.read_csv("data/train.csv"))

X = train_ds.iloc[:,2:9]
y = train_ds.Survived.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

np.random.seed(88)

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
