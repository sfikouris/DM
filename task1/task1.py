import pandas as pd 
import numpy as np

#df = pd.read_csv('data.csv')
df = pd.read_csv('ODI-2021.csv')
#df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'])

#cleaning 
df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'], errors='coerce')
df['Time you went to be Yesterday'] = pd.to_datetime(df['Time you went to be Yesterday'], errors='coerce')

print(df['Time you went to be Yesterday'].dt.time.to_string())

#make the values of column 1 same
df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["ai","AI","Ai","AI (uva)","AI Masters",
    "AI: cognitive science track","artificial intelligence","Artificial Intellingence",
    "MSc AI","Master Artificial Intelligence","Master AI at UvA","Artificial Intelligence","Msc Artificial Intelligence ",
    "artificial intelligence","Artificial Intelligence Masters","Artificial Intelligence MSc","MSc: Artificial Intelligence",
    "Master Artificial Intelligence: Cognitive Sciences ","Masters of Artificial Intelligence","Msc AI","MSc Artificial Intelligence",
    "Msc Artificial Intelligence","MSc Artificial Intelligence @UvA","Artificial intelligence"],
    value = "Artificial Intelligence")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["BA","Business analytics ","Business Analytics & AI","Master Business Analytics"],
    value = "Business Analytics")


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


# cleaning You can get 100 euros if you win a local DM competition, or we don’t hold
#  any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? 
counter = 0 
for row in df['You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? ']:
    try:
        row = float(row)
        if row > 100:
            df.loc[counter,'You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? ']=100
        elif row < 0:
            df.loc[counter,'You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? ']=0
    except ValueError:
        df.loc[counter,'You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? '] =np.nan
    counter+=1


#fixing types

#df['Have you taken a course on machine learning?'] = df['Have you taken a course on machine learning?'].astype('bool')

#print(df['When is your birthday (date)?'].to_string())
#print(df.info())