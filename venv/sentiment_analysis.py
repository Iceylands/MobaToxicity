from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd

#loading the model and tokenizer

roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative','Neutral','Positive']


#preprocess tweet
dota_tweets = pd.read_csv('tweets_dota.csv')

tweet_proc_dota = []

for line in dota_tweets['Tweet']:
    tweet_words = []
    for word in line.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)
    joined = " ".join(tweet_words)
    tweet_proc_dota.append(joined)

#Analysis

for tweet in tweet_proc_dota:
    encoded_tweet = tokenizer(tweet, return_tensors = 'pt')
    #print(encoded_tweet)
    output = model(**encoded_tweet)
    #print(output)
    scores =output[0][0].detach().numpy()
    scores =softmax(scores)
    print("Tweet: ", tweet)
    for i in range(len(scores)):
        l = labels[i]
        s = scores[i]
        print(l,s)

