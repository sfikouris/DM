import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#uncomment the plt.show() function to display the charts.

df = pd.read_csv('test1.csv')

null_val=df.isnull().sum(axis = 0)
print(null_val)

null_val.plot(kind="barh")
plt.tight_layout()
plt.show()


df_tmp = df['What is your gender?']

all_males = df_tmp.value_counts()[0]
all_females = df_tmp.value_counts()[1]
all_unk = df_tmp.value_counts()[2]
male=0
female=0
ukn=0
for index, row in df.iterrows():
    if(pd.isnull(row['When is your birthday (date)?'])):
        if row['What is your gender?'] == "male":
            male+=1
        elif row['What is your gender?'] == "female": 
            female+=1
        else:
            ukn+=1       



x = np.array(["all_males", "null_birthday", "all_females", "null_birthdays"])
y = np.array([all_males, male, all_females, female])
data=[['all_males',all_males], ['null_birthday',male],['all_female',all_females],['null_birthday',female]]
df_gender_nulldate = pd.DataFrame(data,columns = ['Gender', 'Counts'])
#print(df_gender_nulldate)
plt.bar(x, y)
#plt.show()


