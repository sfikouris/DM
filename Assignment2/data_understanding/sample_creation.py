import os
os.chdir("../..")

file = open("Assignment2/data/training_set_VU_DM.csv")
sample = open("Assignment2/data/training_set_sample_1000.csv", "w")

current_line = file.readline()
sample.write(current_line)

search_count = 0
prev_search_id = None
while True:
    current_line = file.readline()
    current_search_id = int(current_line.split(",")[0])

    if current_search_id != prev_search_id:
        search_count += 1
        if search_count > 1000:
            break

    prev_search_id = current_search_id
    sample.write(current_line)

file.close()
sample.close()