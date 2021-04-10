import pandas as pd 
import numpy as np

#df = pd.read_csv('data.csv')
df = pd.read_csv('ODI-2021.csv')
#df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'])

#cleaning 
df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'], errors='coerce')

#cleaning 'what is your stress level (0-100)?' column.
counter = 0
for row in df['What is your stress level (0-100)?']:
    try:
        row = float(row)
        if row > 100:
            df.loc[counter,'What is your stress level (0-100)?']=100
        elif row < 0:
            df.loc[counter,'What is your stress level (0-100)?']=0
    except ValueError:
        df.loc[counter,'What is your stress level (0-100)?']=np.nan
    counter+=1

#print(df['What is your stress level (0-100)?'].to_string())
#print(df.info())

# cleaning Number of neighbors sitting around you? column.
counter = 0
for row in df['Number of neighbors sitting around you?']:
    try:
        row=int(row)
        if row > 10:
            df.loc[counter, 'Number of neighbors sitting around you?']=np.nan
        elif row < 0:
            df.loc[counter, 'Number of neighbors sitting around you?']=np.nan
    except ValueError:
        df.loc[counter, 'Number of neighbors sitting around you?']=np.nan
    counter+=1

print(df['Number of neighbors sitting around you?'].to_string())
#fixing types

#df['Have you taken a course on machine learning?'] = df['Have you taken a course on machine learning?'].astype('bool')

#print(df['When is your birthday (date)?'].to_string())
#print(df.info())