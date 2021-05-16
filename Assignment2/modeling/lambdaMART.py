import pyltr
import os
import numpy as np
import pandas as pd
import sys
from Assignment2.data_preparation.preparation_functions import *


def train_lambdamart(df):
    # convert file data to feature vector, target values and query ids
    feature_vectors, search_ids, _, target_values = prepare_data(df)

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
def prepare_data(df, scoring_type="pw"):

    #perform the feature engineering
    '''
    df = split_date_time(df)
    df = prepare_competitor_data(df)
    df = prepare_countries_id(df)
    df = add_avg_and_normalized_offset(df, ["prop_starrating", 'prop_review_score', "prop_location_score1",
                                            "prop_location_score2", "price_usd", "orig_destination_distance"])
    df = df.drop(['site_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_destination_id',
                  'srch_query_affinity_score', "random_bool"], axis=1)
    '''

    #add target values
    if 'booking_bool' in df.columns:
        if scoring_type == "pw":
            df = add_target_values_pointwise(df)
        elif scoring_type == "lw":
            df = add_target_values_listwise(df)
        elif scoring_type == "hb":
            df = add_target_values_hybrid(df)
        else:
            raise ValueError("unrecognized scoring type")
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

    #Make sure the feature vectors are floats, and replace NULL by 0, so lambdaMART does not complain
    feature_vectors = feature_vectors.apply(pd.to_numeric, errors='coerce', downcast='float')
    feature_vectors = feature_vectors.fillna(float(0))

    return feature_vectors, search_ids, property_ids, target_values


def cross_validation(df, group_amounts):
    searches = df.groupby("srch_id")
    search_groups = list()

    search_groups_filled = False
    i = 0
    for srch_id, srch in searches:
        if not search_groups_filled:
            search_groups.append(srch)
        else:
            search_groups[i] = pd.concat([search_groups[i], srch])

        if i == group_amounts-1:
            search_groups_filled = True
        i = (i+1) % group_amounts

    results = list()

    for i, search_group in enumerate(search_groups):
        validation_set = search_group
        training_set = search_groups.copy()
        del training_set[i]
        training_set = pd.concat(training_set)

        training_feature_vectors, training_search_ids, _, training_target_values = prepare_data(training_set, scoring_type='pw')
        validation_feature_vectors, validation_search_ids, _, validation_target_values = prepare_data(validation_set, scoring_type='pw')

        metric = pyltr.metrics.NDCG()
        monitor = pyltr.models.monitors.ValidationMonitor(validation_feature_vectors,
                                                          validation_target_values,
                                                          validation_search_ids,
                                                          metric=metric,
                                                          trim_on_stop=False)

        model = pyltr.models.LambdaMART(verbose=1,
                                        n_estimators=1000)

        model.fit(training_feature_vectors,
                  training_target_values,
                  training_search_ids,
                  monitor=monitor)

    return results


# testing
os.chdir("../../Assignment2/data")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(threshold=sys.maxsize)


df1 = pd.read_csv("training_set_feature_engineering_done.csv")
#df2 = pd.read_csv("test_set_feature_engineering_done.csv")

cross_validation(df1, 5)

#model = train_lambdamart(df1)
#predict_ranking(model, df2, "lambdaMART_training_test_featureEngineering_hybrid.csv")
