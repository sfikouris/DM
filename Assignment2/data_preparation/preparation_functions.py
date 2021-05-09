import pandas as pd
import os
import numpy as np
from datetime import date
import sys

#read in a single search and return as pandas dataframe, assumes first position contains id
def read_search(file, header):
    search_matrix = []
    prev_file_loc = file.tell()
    curr_line = file.readline().rstrip("\n")
    prev_id = None

    while curr_line:
        data_row = curr_line.split(",")
        curr_id = int(data_row[0])

        #unread the current line if it's not part of the search
        if prev_id != None and prev_id != curr_id:
            file.seek(prev_file_loc, 0)
            break

        search_matrix.append(data_row)
        prev_id = curr_id
        prev_file_loc = file.tell()
        curr_line = file.readline().rstrip("\n")

    output_df = pd.DataFrame(search_matrix, columns=header)
    return output_df

#return the hotels in a single search in the order of likelihood to be booked
def order_hotels(search):
    remaining_hotels = search.copy()
    #first, put the sample with booked_bool at the top
    booked_hotels = remaining_hotels.loc[remaining_hotels['booking_bool'] == "1"]
    remaining_hotels = pd.concat([search, booked_hotels]).drop_duplicates(keep=False)

    #second, choose the samples with the clicked_bool
    clicked_hotels = remaining_hotels.loc[remaining_hotels['click_bool'] == "1"]
    remaining_hotels = pd.concat([remaining_hotels, clicked_hotels]).drop_duplicates(keep=False)

    #third, put the remaining in based upon position
    remaining_hotels['position'] = remaining_hotels['position'].apply(pd.to_numeric)
    remaining_hotels = remaining_hotels.sort_values(by=['position'])

    result = pd.concat([booked_hotels, clicked_hotels, remaining_hotels])
    return result

#adds the same target values as used in the competition. 5 for book, 1 for click.
def add_target_values_pointwise(df):
    df['target_value'] = df['booking_bool'] * 4 + df['click_bool']
    return df

#add target values, ordering based upon booked, clicked and then position
def add_target_values_listwise(df):
    searches = df.groupby("srch_id")
    for id, search in searches:
        max_pos = search['position'].max()
        search['position'] = -(search['booking_bool'] + search['click_bool']) * max_pos + search['position']
        search = search.sort_values(by=['position'])
        search_size = len(search.index)
        i = 5
        x = 5/search_size
        for index, row in search.iterrows():
            df.at[index, 'target_value'] = i
            i -= x
    return df

#splits the date_time column into 2 new columns date and time, which use the range [0,1]
def split_date_time(df):
    for index, row in df.iterrows():
        date_time_str = row['date_time']
        date_str, time_str = date_time_str.split(' ')
        year, month, day = map(int, date_str.split('-'))
        hour, minute, second = map(int,time_str.split(':'))

        year_start = date(year, 1, 1)
        search_date = date(year, month, day)
        date_delta = search_date - year_start
        date_delta_norm = date_delta.days / 365

        hour_delta_norm = hour / 24
        minute_delta_norm = minute / 60 / 24
        second_delta_norm = second / 60 / 60 / 24
        time_delta_norm = hour_delta_norm + minute_delta_norm + second_delta_norm

        df.at[index, 'date'] = date_delta_norm
        df.at[index, 'time'] = time_delta_norm

    df = df.drop('date_time', axis=1)
    return df

#prepares data about competitor pricing, combining all competitors. Quite time intensive
def prepare_competitor_data(df):
    for index, row in df.iterrows():
        comp_rate_worse = 0
        comp_rate_better = 0
        comp_inv_true = 0
        comp_inv_false = 0
        comp_total_higher_rate = 0
        comp_num_higher_rate = 0
        comp_total_lower_rate = 0
        comp_num_lower_rate = 0

        for i in range(1, 9):
            if row['comp{}_rate'.format(i)] == 1.0:
                comp_rate_worse += 1

                if not pd.isna('comp{}_rate_percent_diff'.format(i)):
                    comp_rate = row['comp{}_rate_percent_diff'.format(i)]
                    #cap outliers at 200
                    if comp_rate > 200:
                        comp_rate = 200
                    comp_total_higher_rate += comp_rate
                    comp_num_higher_rate += 1

            elif row['comp{}_rate'.format(i)] == -1.0:
                comp_rate_better += 1

                if not pd.isna(row['comp{}_rate_percent_diff'.format(i)]):
                    comp_rate = row['comp{}_rate_percent_diff'.format(i)]
                    #cap outliers at 200
                    if comp_rate > 200:
                        comp_rate = 200
                    comp_total_lower_rate += comp_rate
                    comp_num_lower_rate += 1

            if row['comp{}_inv'.format(i)] == 0:
                comp_inv_true += 1
            elif row['comp{}_inv'.format(i)] == 1:
                comp_inv_false += 1

        if comp_num_higher_rate > 0:
            comp_mean_higher_rate = comp_total_higher_rate / comp_num_higher_rate
        else:
            comp_mean_higher_rate = 0

        if comp_num_lower_rate > 0:
            comp_mean_lower_rate = comp_total_lower_rate / comp_num_lower_rate
        else:
            comp_mean_lower_rate = 0

        df.at[index, 'comp_rate_worse'] = comp_rate_worse
        df.at[index, 'comp_rate_better'] = comp_rate_better
        df.at[index, 'comp_inv_true'] = comp_inv_true
        df.at[index, 'comp_inv_false'] = comp_inv_false
        df.at[index, 'comp_mean_higher_rate'] = comp_mean_higher_rate
        df.at[index, 'comp_mean_lower_rate'] = comp_mean_lower_rate

    for i in range(1, 9):
        df = df.drop(['comp{}_rate'.format(i), 'comp{}_inv'.format(i), 'comp{}_rate_percent_diff'.format(i)], axis=1)
    return df


#prepares data about avg price per srch_id
def prepare_price_data(df):
    srch_grp = df.groupby(['srch_id'])
    AvgHotelPricePerSrchID = srch_grp['price_usd'].mean()
    df_newbook=df.rename(columns={'price_usd': 'HotelPrice'})
    PriceOfHotel=df_newbook.set_index('srch_id')['HotelPrice']
    python_df = pd.concat([AvgHotelPricePerSrchID, PriceOfHotel], axis='columns', sort= False)
    python_df['normalize']=1

    a, b = -1,1

    df1 = python_df.groupby(python_df.index)
    index = python_df.index
    uni_ind = index.unique()
    for ind in uni_ind:
        pre_norm = df1.get_group(ind).price_usd - df1.get_group(ind).HotelPrice
        x, y = pre_norm.min(), pre_norm.max()
        python_df.loc[ind,'normalize'] = (pre_norm - x) / (y-x) * (b-a) + a

#prepares data about location score 1 per srch_id
def prepare_location1_score(df):
    df_loc1score=df.rename(columns={'prop_location_score1': 'AvgLocation_score1'})
    loc1_grp = df_loc1score.groupby(['srch_id'])
    AvgScoreLoc1 = loc1_grp['AvgLocation_score1'].mean()
    Loc1_score = df.set_index('srch_id')['prop_location_score1']
    df_loc1 = pd.concat([AvgScoreLoc1, Loc1_score], axis='columns', sort= False)
    df_loc1['normalize']=1
    df_loc1['pre_norm'] =1 
    a, b = -1,1
    df_loc1_new = df_loc1.groupby(df_loc1.index)
    index = df_loc1.index
    uni_ind = index.unique()
    for ind in uni_ind:
        pre_norm = df_loc1_new.get_group(ind).AvgLocation_score1 - df_loc1_new.get_group(ind).prop_location_score1
        df_loc1.loc[ind,'pre_norm'] = pre_norm
        x, y = pre_norm.min(), pre_norm.max()
        df_loc1.loc[ind,'normalize'] = (pre_norm - x) / (y-x) * (b-a) + a

#prepares data about location score 2 per srch_id
def prepare_location2_score(df):
    df_loc2score=df.rename(columns={'prop_location_score2': 'AvgLocation_score2'})
    loc2_grp = df_loc2score.groupby(['srch_id'])
    AvgScoreLoc2 = loc2_grp['AvgLocation_score2'].mean()
    Loc2_score = df.set_index('srch_id')['prop_location_score2']
    df_loc2 = pd.concat([AvgScoreLoc2, Loc2_score], axis='columns', sort= False)
    df_loc2['normalize']=1
    df_loc2['pre_norm'] =1 
    a, b = -1,1
    df_loc2_new = df_loc2.groupby(df_loc2.index)
    index = df_loc2.index
    uni_ind = index.unique()
    for ind in uni_ind:
        pre_norm = df_loc2_new.get_group(ind).AvgLocation_score2 - df_loc2_new.get_group(ind).prop_location_score2
        df_loc2.loc[ind,'pre_norm'] = pre_norm
        x, y = pre_norm.min(), pre_norm.max()
        df_loc2.loc[ind,'normalize'] = (pre_norm - x) / (y-x) * (b-a) + a

#prepares data about orig_destination_distance per srch_id
def prepare_orig_destination_distance(df):
    df_dest=df.rename(columns={'orig_destination_distance': 'AvgDestDist'})
    dist_grp = df_dest.groupby(['srch_id'])
    AvgDestDist = dist_grp['AvgDestDist'].mean()
    df_dest1 = df.set_index('srch_id')['orig_destination_distance']
    dest = pd.concat([AvgDestDist, df_dest1], axis='columns', sort= False)
    dest['normalize']=1
    dest['pre_norm'] =1 

    a, b = -1,1
    df_dest_new = dest.groupby(dest.index)
    index = dest.index
    uni_ind = index.unique()
    for ind in uni_ind:
        pre_norm = df_dest_new.get_group(ind).AvgDestDist - df_dest_new.get_group(ind).orig_destination_distance
        dest.loc[ind,'pre_norm'] = pre_norm
        x, y = pre_norm.min(), pre_norm.max()
        dest.loc[ind,'normalize'] = (pre_norm - x) / (y-x) * (b-a) + a

def prepare_countries_id(df):
    df['same_country'] = (df['visitor_location_country_id'] == df['prop_country_id'])
    df.same_country = df.same_country.astype(int)
    return df.drop(['visitor_location_country_id','prop_country_id'],axis=1)