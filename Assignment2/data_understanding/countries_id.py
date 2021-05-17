from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython


chunksize = 10 ** 6
same = pd.DataFrame()
diff = pd.DataFrame()
orig_distance = pd.DataFrame()
nancntr = 0
for chunk in pd.read_csv("../../as2/data/training_set_VU_DM.csv", chunksize=chunksize):
    same_country_id = (chunk['visitor_location_country_id'] == chunk['prop_country_id'])
    booked = (chunk['booking_bool'] == 1)
    same = same.append(chunk[same_country_id & booked])
    diff = diff.append(chunk[~same_country_id & booked])
    orig_distance = orig_distance.append(chunk[booked].fillna({'orig_destination_distance' : 0}))
    nancntr += chunk[booked].orig_destination_distance.isna().sum()
    # if(max_price < chunk.loc[ booked, 'gross_bookings_usd' ].max()):
    #     max_price = chunk.loc[ booked, 'gross_bookings_usd' ].max()

IPython.embed()

def plot_hist(data,title,color):
    bins = np.arange(0,2000,50)

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(data, bins, color=color, density=True)
    
    ax.set_xlabel(title)
    fig.tight_layout()
    plt.show()

# plot_hist(diff.gross_bookings_usd,'Booked Prices','blue')
# plot_hist(same.gross_bookings_usd,'Booked Prices','orange')

orig_distance = orig_distance.sort_values('orig_destination_distance')
length = int(len(orig_distance)*0.25)
plot_hist(orig_distance[-length:].gross_bookings_usd,'Longer Distance prices','blue')
plot_hist(orig_distance[:length].gross_bookings_usd, 'Closer Distance prices','orange')
