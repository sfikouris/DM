from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython


df = pd.read_csv("../data/training_set_sample_1000.csv")

same_country_id = (df['visitor_location_country_id'] == df['prop_country_id'])
booked = (df['booking_bool'] == 1)

same = df[same_country_id & booked] 
diff = df[~same_country_id & booked]

orig_distance = df[booked].fillna({'orig_destination_distance' : 0}).sort_values('orig_destination_distance')

IPython.embed()

def plot_hist(data):
    bins = np.arange(0,df.loc[ booked, 'price_usd' ].max(),100)

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(data, bins, density=True)
    
    ax.set_xlabel('Price')
    fig.tight_layout()
    plt.show()

plot_hist(diff.price_usd)
plot_hist(same.price_usd)

length = int(len(orig_distance)*0.25)
plot_hist(orig_distance[-length:].price_usd)
plot_hist(orig_distance[:length].price_usd)
