import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
import operator
import math

with open("C:\\Users\\Casper\\PycharmProjects\\DM\\task3\\3C\\SmsCollection.csv") as myfile:
    mydata = [line.strip() for line in myfile]
    SMSColl = pd.Series(mydata)

SMSColl = SMSColl.str.split(pat=";", n=1, expand=True)
SMSColl = SMSColl.rename(columns=SMSColl.iloc[0]).drop(SMSColl.index[0])

#tokenize
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
SMSColl['text'] = SMSColl.apply(lambda row: tokenizer.tokenize(row['text']), axis=1)

#spellchecking
#for text in SMSColl['text']:
#    i=0
#    for word in text:
#        textBlb = TextBlob(word)  # Making our first textblob
#        wordCorrected = textBlb.correct()  # Correcting the text
#        word = wordCorrected
#        print(word)
#        i += 1
#    print(text)

#normalize/lemmatize
lemmatiser = WordNetLemmatizer()
SMSColl['text'] = SMSColl.apply(lambda row: [lemmatiser.lemmatize(token.lower(), pos='v') for token in row['text']], axis=1)

#remove stop words
SMSColl['text'] = SMSColl.apply(lambda row: [token for token in row['text'] if token not in stopwords.words('english')], axis=1)

#temp splitting training and testing set
train = SMSColl.iloc[:2787]
test = SMSColl.iloc[2787:]
SMSColl = train

#count words
mainCount = Counter()
hamCount = Counter()
spamCount = Counter()

for index, entry in SMSColl.iterrows():
    if entry['label'] == 'ham':
        for word in entry['text']:
            mainCount[word] += 1
            hamCount[word] += 1

    elif entry['label'] == 'spam':
        for word in entry['text']:
            mainCount[word] += 1
            spamCount[word] += 1


mainCount = mainCount.most_common()
mainCount = [i for i in mainCount if i[1] >= 1]

wordDict = {}
for word, count in mainCount:
    wordValue = (hamCount[word] - spamCount[word]) / count
    if wordValue != 0:
        wordDict[word] = wordValue

#wordDict = sorted(wordDict.items(), key=operator.itemgetter(1), reverse=True)
print(wordDict)

#check performance
SMSColl = test
spamCorrect = 0
spamIncorrect = 0
hamCorrect = 0
hamIncorrect = 0

for index, entry in SMSColl.iterrows():
    hamConfidence = 0
    for word in entry['text']:
        if word in wordDict:
            wordValue = wordDict[word]
            hamConfidence += (wordValue ** 2) * (math.sqrt(wordValue ** 2)/wordValue)
    if entry['label'] == 'ham':
        if hamConfidence >= 0:
            hamCorrect += 1
        else:
            hamIncorrect += 1

    elif entry['label'] == 'spam':
        if hamConfidence < 0:
            spamCorrect += 1
        else:
            spamIncorrect += 1


print("Ham correct/incorrect: ", hamCorrect,"/",hamIncorrect)
print("Spam correct/incorrect: ", spamCorrect,"/",spamIncorrect)

#print(Counter([i[1] for i in mainCount]))
#print(SMSColl)
#print(mainCount)
#print(hamCount)
#print(spamCount)

