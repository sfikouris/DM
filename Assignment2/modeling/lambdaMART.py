import pyltr
import os
import numpy as np
import sys
from Assignment2.data_preparation.preparation_functions import *


def train_lambdamart(filename):
    # convert file data to feature vector, target values and query ids
    feature_vectors, search_ids, _, target_values = prepare_data(filename)

    model = pyltr.models.LambdaMART(verbose=1)
    model.fit(feature_vectors, target_values, search_ids)

    return model


def predict_ranking(model, filename_input, filename_output):
    feature_vectors, search_ids, property_ids, _ = prepare_data(filename_input)
    output_file = open(filename_output, "w")
    output_file.write("srch_id,prop_id\n")

    samples = []
    curr_prop_ids = []
    prev_id = search_ids[0]
    # fix issue with the last sample being ignored
    for i, sample in enumerate(feature_vectors):
        curr_id = search_ids[i]
        if prev_id != curr_id:
            write_model_search_prediction(output_file, model, samples, curr_prop_ids, prev_id)
            samples = []
            curr_prop_ids = []

        samples.append(sample)
        curr_prop_ids.append(property_ids[i])
        prev_id = curr_id

    write_model_search_prediction(output_file, model, samples, curr_prop_ids, curr_id)

    output_file.close()


def write_model_search_prediction(file, model, search, curr_prop_ids, search_id):
    prediction_values = model.predict(search)
    prediction_amount = len(prediction_values)
    prediction_order = np.argsort(prediction_values)

    # get the order of the property by their prediction value. Then print to the output file
    property_order = np.zeros(prediction_amount)

    for j, prop_id in enumerate(curr_prop_ids):
        prop_order_loc = prediction_order[j]
        property_order[prop_order_loc] = prop_id

    for prop_id in property_order:
        file.write(str(search_id) + "," + str(int(prop_id)) + "\n")


# prepare the data and return the feature vectors, property ids and search ids, and target values if possible
def prepare_data(filename):
    file = open(filename)

    header_string = file.readline().rstrip("\n")
    header = header_string.split(",")

    feature_vectors = []
    target_values = []
    search_ids = []
    property_ids = []

    search = read_search(file, header)

    while not search.empty:
        # if the data contains training data, calculate target values.
        if 'booking_bool' in search.columns:
            search = order_hotels(search)
            search = search.drop(['position', 'click_bool', 'booking_bool', 'gross_bookings_usd'], axis=1)

            sample_amount = len(search.index)
            target_values = target_values + list(range(sample_amount))

        # split off the search ids and propery ids
        search_ids = search_ids + search['srch_id'].tolist()
        search = search.drop(['srch_id'], axis=1)

        property_ids = property_ids + search['prop_id'].tolist()
        search = search.drop(['prop_id'], axis=1)

        # forced feature engineering, can't deal with categorical and NULL
        search = search.apply(pd.to_numeric, errors='coerce', downcast='float')
        search = search.fillna(float(0))

        # get the feature vectors
        feature_vectors = feature_vectors + search.values.tolist()

        # get the next search
        search = read_search(file, header)
        print("current search processed: ", search_ids[-1])

    # turn lists into numpy arrays
    search_ids = np.array(search_ids)
    target_values = np.array(target_values)
    property_ids = np.array(property_ids)
    X = np.ndarray((len(feature_vectors), len(feature_vectors[0])), dtype=np.float64)
    for i, x in enumerate(feature_vectors):
        X[i] = x
    feature_vectors = X

    search_ids = search_ids.astype(int)
    target_values = target_values.astype(float)
    property_ids = property_ids.astype(int)

    file.close()
    return feature_vectors, search_ids, property_ids, target_values


# testing
os.chdir("../..")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
np.set_printoptions(threshold=sys.maxsize)

model = train_lambdamart("Assignment2/data/training_set_VU_DM.csv")
predict_ranking(model, "Assignment2/data/test_set_VU_DM.csv",
                "Assignment2/data/lambdaMART_train_train_test_test.csv")
