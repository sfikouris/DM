import pandas as pd 

#df = pd.read_csv('data.csv')
df = pd.read_csv('ODI-2021.csv')
#df['column'] = df['column'].astype('|S')
#df['column_name'] = df['column_name'].astype('bool')
#df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'])
df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'], errors='coerce')
#print(df['When is your birthday (date)?'].to_string())
print(df.info())