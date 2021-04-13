import pandas as pd 
import numpy as np

#df = pd.read_csv('data.csv')
df = pd.read_csv('ODI-2021.csv')
#df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'])

#cleaning 
df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'], errors='coerce')
df['Time you went to be Yesterday'] = pd.to_datetime(df['Time you went to be Yesterday'], errors='coerce')

#print(df['Time you went to be Yesterday'].dt.time.to_string())



#make the values of column 1 same
df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["ai","AI","Ai","AI (uva)","AI Masters",
    "AI: cognitive science track","artificial intelligence","Artificial Intellingence",
    "MSc AI","Master Artificial Intelligence","Master AI at UvA","Artificial Intelligence","Msc Artificial Intelligence ",
    "artificial intelligence","Artificial Intelligence Masters","Artificial Intelligence MSc","MSc: Artificial Intelligence",
    "Master Artificial Intelligence: Cognitive Sciences ","Masters of Artificial Intelligence","Msc AI","MSc Artificial Intelligence",
    "Msc Artificial Intelligence","MSc Artificial Intelligence @UvA","Artificial intelligence","M AI","MSc Ai"],
    value = "Artificial Intelligence")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["BA","Business analytics ","Business Analytics & AI","Master Business Analytics"],
    value = "Business Analytics")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["Bioinformatics","Bioinformatics ","Bioinformatics & Systems biology","Bioinfromatics","Master Bioinformatics and Systems Biology",
    "Bioinformatics & Systems Biology","bioinformatics and system biology","Bioinformatics and systems biology",
    "Master bioinformatics and systems biology","Master Bioinformatics & Systems Biology",
    "MSc Bioinformatics & Systems Biology","MSc Bioinformatics and systems biology","MSC Bioinformatics and systems Biology",
    "MSc in Bioinformatics and Systems Biology","Bioinformatics and systems biology ","M Bioinformatics and Systems Biology"],
    value = "Bioinformatics and Systems Biology")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["computational science","Computational science","Computational Science UvA/VU",
    "Master in Computational Science","Computational Sciences","Computational Sciences "],
    value = "Computational Science")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["CS: Big Data","big data engineering "],
    value = "Big Data")    

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["Climate econometrics"],
    value = "Climate Econometrics")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["Data science","Data Science","Data Science (information studies)","Data Science track (UvA)",
    "Information sciences","Information Studies (Data Science) @ UvA","Information studies Data Science (UvA)",
    "Information Studies: Data Science (track)","M Information Sciences","Master Information Studies UvA","MSc Information studies"],
    value = "Information Studies: Data Science")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["Econometrics and data science","Econometrics and data science ","Econometrics and Data Science","EDS"],
    value = "Econometrics & Data Science")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["Econometrics & Operation Research master","Econometrics and Operation Research","Econometrics and Operations Research",
    "EOR","Master Econometrics and Operations Research: Data Science track","MSc Econometrics and Operation Research",
    "Msc Econometrics and Operations Research","E&OR","Econometrics and Operations Research "],
    value = "Econometrics & Operations Research")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["QRM","Duisenberg Honours Programme in Finance and Technology","Duisenberg Honours: QRM",
    "MSc Finance - Duisenberg Honours Programme of Quantitative Risk Management","Quantitative Risk Management","Quantitative Risk Management "],
    value = "Quantative Risk Management")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["Finance & Technology","Finance and technology","Fintech","Honours master F&T","MSc Finance",
    "Duisenberg Honours Programme in Finance and Technology","Master Finance & Technology","Msc. Finance and Technology"],
    value = "Finance and Technology")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["HLT","Human Language Technology (Research Master Linguistics)","rMA Human Language Technology",
    "RMA Human Language Technology","Research Masters Human Language Technology"],
    value = "Human Language Technology")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["Linguistics: Text Mining"],
    value = "Linguistics Text Mining")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["Master Artificial Intelligence: Cognitive Sciences ","RM AI, RM Cognitive Neuropsychology","AI: cognitive science track"],
    value = "Cognitive Sciences")

df['What programme are you in?'] = df['What programme are you in?'].replace(
    to_replace = ["cs","CS","Msc Computer Science"],
    value = "Computer Science")

df.loc[df['What programme are you in?']=='https://forms.gle/eTy4nEs3khRqPtMLA','What programme are you in?'] = np.nan
df.loc[df['What programme are you in?']=='Econometrics','What programme are you in?'] = np.nan
df.loc[df['What programme are you in?']=='Econometrics ','What programme are you in?'] = np.nan
df.loc[df['What programme are you in?']=='OR','What programme are you in?'] = np.nan
df.loc[df['What programme are you in?']=='Je zusje','What programme are you in?'] = np.nan
df.loc[df['What programme are you in?']=='Python','What programme are you in?'] = np.nan



#print(df['What programme are you in?'].to_string())

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


df.to_csv('test1.csv')

#fixing types

#df['Have you taken a course on machine learning?'] = df['Have you taken a course on machine learning?'].astype('bool')

#print(df['When is your birthday (date)?'].to_string())
#print(df.info())