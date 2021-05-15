import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def randomforest(filename):
    df = pd.read_csv(filename)

    df = df.drop(['date_time'], axis=1)

    df['n_rating'] = 0
    df['n_rating'] = np.where( (df['click_bool']==1)  & (df['booking_bool']==1), 8, df['n_rating'] )
    df['n_rating'] = np.where( (df['click_bool']==1)  & (df['booking_bool']==0), 3, df['n_rating'] )
    df['n_rating'] = np.where( (df['click_bool']==0)  & (df['booking_bool']==1), 5, df['n_rating'] )
    df['n_rating'] = np.where( (df['click_bool']==0)  & (df['booking_bool']==0), 1, df['n_rating'] )

    X = df.drop(['n_rating', 'click_bool', 'booking_bool'], axis = 1)
    y = df['n_rating']

    training, testing, training_labels, testing_labels = train_test_split(X, y, test_size = .25, random_state = 42)



    clf = RandomForestClassifier()
    clf.fit(training, training_labels)
    preds = clf.predict(testing)
    print (clf.score(training, training_labels))
    print(clf.score(testing, testing_labels))

randomforest('../../Assignment2/data/training_set_sample_1000.csv')