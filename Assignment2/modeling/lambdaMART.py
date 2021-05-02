import pyltr
import os
import numpy as np
import sys
from Assignment2.data_preparation.test_file import *

def train_lambdamart(filename):
    #convert file data to training vector, target values and query ids
    file = open(filename)

    header_string = file.readline().rstrip("\n")
    header = header_string.split(",")

    training_vectors = []
    target_values = []
    search_ids = []

    search = read_search(file, header)

    while not search.empty:
        search = order_hotels(search)

        #ideally, rethink this and skip this step.
        search = search.drop(['position', 'click_bool', 'booking_bool', 'gross_bookings_usd'], axis=1)

        #this is nonsensical for categorical values. So needs some feature engineering. Also currently can't deal with NULL
        search = search.apply(pd.to_numeric, errors='coerce',downcast='float')
        search = search.fillna(float(0))

        #get the query id
        search_ids = search_ids + search['srch_id'].tolist()
        search = search.drop(['srch_id'], axis=1)

        #get the target values
        sample_amount = len(search.index)
        target_values = target_values + list(range(sample_amount))

        #get the training vectors
        training_vectors = training_vectors + search.values.tolist()

        #get the next search
        search = read_search(file, header)

        print(search_ids[-1])

    #turn lists into numpy arrays
    search_ids = np.array(search_ids)
    target_values = np.array(target_values)
    X = np.ndarray((len(training_vectors), len(training_vectors[0])), dtype=np.float64) #todo rename x
    for i, x in enumerate(training_vectors):
        X[i] = x
    training_vectors = X

    search_ids = search_ids.astype(int)
    target_values = target_values.astype(float)

    model = pyltr.models.LambdaMART(verbose=1)
    model.fit(training_vectors, target_values, search_ids)
    file.close()

    #testing
    metric = pyltr.metrics.dcg.NDCG()
    predictions = model.predict(training_vectors)
    print("random: ", metric.calc_mean_random(search_ids, target_values))
    print("prediction: ", metric.calc_mean(search_ids, target_values, predictions))

    samples = []
    prev_id = search_ids[0]
    for i, sample in enumerate(training_vectors):
        curr_id = search_ids[i]
        if prev_id != curr_id:
            prediction = model.predict(samples)
            print(prev_id, ": ", np.argsort(prediction))
            samples = []
        samples.append(sample)
        prev_id = curr_id

    return model

def predict_ranking(model, filename):
    file = open(filename)
    #todo: read in validation set and perform ranking
    file.close()

#testing
os.chdir("../..")
pd.set_option("display.max_columns",None)
pd.set_option("display.width", None)
np.set_printoptions(threshold=sys.maxsize)
model = train_lambdamart("Assignment2/data/training_set_sample_1000.csv")

print(predict_ranking(model, "Assignment2/data/training_set_sample_1000.csv"))