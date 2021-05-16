from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("../..")

file = open("Assignment2/data/training_set_VU_DM.csv")

#get rid of header
file.readline()

booked_counter = Counter()
clicked_counter = Counter()
position_counter = Counter()

#testing
clicked_amount = 0
booked_amount = 0
#testing

prev_id = None
search_matrix = []
max_pos = 0
for line in file:
    data_array = line.split(",")
    curr_id = int(data_array[0])

    if prev_id != curr_id:
        unused_positions = []
        for possible_position in range(1, max_pos+1):
            is_used = False
            for search_row in search_matrix:
                if search_row[0] == possible_position:
                    is_used = True
            if not is_used:
                unused_positions.append(possible_position)

        for search_row in search_matrix:
            unused_amount_passed = 0
            row_position = search_row[0]
            for unused_position in unused_positions:
                if unused_position < row_position:
                    unused_amount_passed += 1
            search_row[0] = row_position - unused_amount_passed

        for search_row in search_matrix:
            position, booking_bool, click_bool = search_row
            position_counter[position] += 1
            if booking_bool == 1:
                booked_counter[position] += 1
                booked_amount += 1
            if click_bool == 1:
                clicked_counter[position] += 1
                clicked_amount += 1

        search_matrix = []
        max_pos = 0

    #only use if random_bool true
    if int(data_array[26]) == 0:
        continue

    position = int(data_array[14])
    booking_bool = int(data_array[53])
    click_bool = int(data_array[51])
    search_row = [position, booking_bool, click_bool]

    search_matrix.append(search_row)
    if position > max_pos:
        max_pos = position

    prev_id = curr_id

file.close()

booked_keys = list(booked_counter.keys())
clicked_keys = list(clicked_counter.keys())

booked_keys = list(map(int, booked_keys))
clicked_keys = list(map(int, clicked_keys))

booked_keys.sort()
clicked_keys.sort()

booked_values = [100*(booked_counter[booked_key]/position_counter[booked_key]) for booked_key in booked_keys]
clicked_values = [100*(clicked_counter[clicked_key]/position_counter[clicked_key]) for clicked_key in clicked_keys]

print(booked_counter)
print(booked_keys)
print(booked_values)
print()
print(clicked_counter)
print(clicked_keys)
print(clicked_values)
print("booked/clicked percentage = %",100*(booked_amount/clicked_amount))

plt.barh(booked_keys, booked_values)
plt.title("booked distribution", fontsize=13)
plt.ylabel("position", fontsize=12)
plt.xlabel("occurrence (percentage)", fontsize=12)
plt.show()

plt.barh(clicked_keys, clicked_values, color="darkorange")
plt.title("clicked distribution", fontsize = 13)
plt.ylabel("position", fontsize = 12)
plt.xlabel("occurrence (percentage)", fontsize = 12)
plt.show()

