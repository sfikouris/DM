import pandas as pd
import os

#read in a single search and return as pandas dataframe, assumes first position contains id
def read_search(file, header):
    search_matrix = []
    prev_file_loc = file.tell()
    curr_line = file.readline()
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
        curr_line = file.readline()

    output_df = pd.DataFrame(search_matrix, columns=header)
    return output_df


#testing
os.chdir("../..")
file = open("Assignment2/data/training_set_sample_1000.csv")
pd.set_option("display.max_columns",None)
pd.set_option("display.width", None)

header_string = file.readline()
header = header_string.split(",")
df = read_search(file, header)
print(df)
print()
df = read_search(file, header)
print(df)
print()

file.close()
