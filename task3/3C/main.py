import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

with open("C:\\Users\\Casper\\PycharmProjects\\DM\\task3\\3C\\SmsCollection.csv") as myfile:
    mydata = [line.strip() for line in myfile]
    SMSColl = pd.Series(mydata)

SMSColl = SMSColl.str.split(pat=";", n=1, expand=True)
SMSColl = SMSColl.rename(columns=SMSColl.iloc[0]).drop(SMSColl.index[0])

#tokenize
SMSColl['text'] = SMSColl.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)

#normalize/lemmatize
lemmatiser = WordNetLemmatizer()
SMSColl['text'] = SMSColl.apply(lambda row: [lemmatiser.lemmatize(token.lower(), pos='v') for token in row['text']], axis=1)

#remove stop words
SMSColl['text'] = SMSColl.apply(lambda row: [token for token in row['text'] if token not in stopwords.words('english')], axis=1)


print(SMSColl)


