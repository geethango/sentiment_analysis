import nltk
import pandas as pd

#Reading File
df=pd.read_csv(r"C:/Users/geeth/Desktop/sentiments data.csv")
textarray = df['SentimentText']

from nltk.corpus import stopwords   
import numpy as np
import re
 
#Prepare the dataset for training
def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) 
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    stop_words = set(stopwords.words('english'))
    temp = [w for w in temp if not w in stop_words]
    temp = " ".join(word for word in temp)
    return temp
results = [clean_tweet(tw) for tw in textarray]
df['cleaned_words']=results

#training 
from sklearn.model_selection import train_test_split
train,test = train_test_split(df,test_size=0.3)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(binary=True, min_df = 10, max_df = 0.95)
cv.fit_transform(train['cleaned_words'].values)

train_x=cv.transform(train['cleaned_words'].values)
test_x=cv.transform(test['cleaned_words'].values)

train_y=train['Sentiment']
test_y=test['Sentiment']

#model fitting
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = 'liblinear', random_state = 50, max_iter=1000)
lr.fit(train_x,train_y)
pred_y = lr.predict(test_x)

#testing accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
print("Accuracy: ",round(accuracy_score(test_y,pred_y),3))
print("F1: ",round(f1_score(test_y, pred_y),3))

#Process user input data
while True:
   input_text =[input("Enter a text:")]
   test_x2=cv.transform(input_text)
   pred_y2=lr.predict(test_x2)
   print(pred_y2)
   if(pred_y2==0):
       print("Sad")
   else:
       print("Happy")
          
   if input("Do You Want To Continue? [y/n]") != "y":
       break
