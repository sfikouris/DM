import pandas as pd
import os
import numpy as np

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

'''
#testing
os.chdir("../..")
file = open("Assignment2/data/training_set_sample_1000.csv")
pd.set_option("display.max_columns",None)
pd.set_option("display.width", None)

header_string = file.readline().rstrip("\n")
header = header_string.split(",")

for i in range(1):
    df = read_search(file, header)
    df = order_hotels(df)
    print(df['srch_id'].tolist())
    print(df)

    sample_amount = len(df.index)
    print(list(range(sample_amount)))


file.close()
'''