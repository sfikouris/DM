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

    df = feature_vectors
    df['srch_id'] = search_ids
    df['prop_id'] = property_ids
    searches = df.groupby("srch_id")
    for id, search in searches:
        search_features = search.drop(['srch_id','prop_id'], axis=1)
        prediction_values = model.predict(search_features)
        prediction_amount = len(prediction_values)
        prediction_order = np.argsort(prediction_values)
        property_order = np.zeros(prediction_amount)
        curr_prop_ids = search['prop_id'].to_numpy()

        for i in range(prediction_amount):
            property_order[i] = curr_prop_ids[prediction_order[i]]

        for prop_id in reversed(property_order):
            output_file.write(str(id) + "," + str(int(prop_id)) + "\n")

    output_file.close()

# prepare the data and return the feature vectors, property ids and search ids, and target values if possible
def prepare_data(filename):
    df = pd.read_csv(filename)

    #add target values
    if 'booking_bool' in df.columns:
        #df = add_target_values_pointwise(df) #todo calc pointwise
        df = add_target_values_listwise(df)
        df = df.drop(['position', 'click_bool', 'booking_bool', 'gross_bookings_usd'], axis=1)
        target_values = df['target_value']
    else:
        target_values = []

    search_ids = df['srch_id']
    property_ids = df['prop_id']
    feature_vectors = df.drop(['srch_id', 'prop_id', 'target_value'], axis=1, errors='ignore')

    search_ids = pd.to_numeric(search_ids, downcast= 'integer')
    property_ids = pd.to_numeric(property_ids, downcast= 'integer')
    target_values = pd.to_numeric(target_values, downcast= 'float')

    # forced feature engineering, can't deal with categorical and NULL
    feature_vectors = feature_vectors.apply(pd.to_numeric, errors='coerce', downcast='float')
    feature_vectors = feature_vectors.fillna(float(0))

    return feature_vectors, search_ids, property_ids, target_values


# testing
os.chdir("../..")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option('display.max_rows', None)

np.set_printoptions(threshold=sys.maxsize)

model = train_lambdamart("Assignment2/data/training_set_VU_DM.csv")
predict_ranking(model, "Assignment2/data/training_set_sample_1000.csv", "Assignment2/data/lambdaMART_trSample_trSample_listwise.csv")
