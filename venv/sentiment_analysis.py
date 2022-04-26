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
league_tweets = pd.read_csv('tweets_lol.csv')

tweet_proc_dota = []
tweet_proc_lol = []

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

for line in league_tweets['Tweet']:
    tweet_words = []
    for word in line.split(' '):
        if word.startswith('@') and len(word) > 1:
             word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)
    joined = " ".join(tweet_words)
    tweet_proc_lol.append(joined)

#Analysis
feelings_dota = []
feelings_lol = []

print("Dota 2")
for tweet in tweet_proc_dota:
    encoded_tweet = tokenizer(tweet, return_tensors = 'pt')
    #print(encoded_tweet)
    output = model(**encoded_tweet)
    #print(output)
    scores =output[0][0].detach().numpy()
    scores =softmax(scores)
    mostfeel = 0.0
    currentfeeling = 'Positive'
    for i in range(len(scores)):
        if scores[i] > mostfeel:
            mostfeel = scores[i]
            currentfeeling = labels[i]
    feelings_dota.append(currentfeeling)
print('Dota 2 feelings for Positive, Neutral, and Negative', feelings_dota.count('Positive'), feelings_dota.count('Neutral'), feelings_dota.count('Negative'))

print("League of Legends")
for tweet in tweet_proc_lol:
    encoded_tweet = tokenizer(tweet, return_tensors = 'pt')
    #print(encoded_tweet)
    output = model(**encoded_tweet)
    #print(output)
    scores =output[0][0].detach().numpy()
    scores =softmax(scores)
    mostfeel = 0.0
    currentfeeling = 'Positive'
    for i in range(len(scores)):
        if scores[i] > mostfeel:
            mostfeel = scores[i]
            currentfeeling = labels[i]
    feelings_lol.append(currentfeeling)
print('League feelings for Positive, Neutral, and Negative', feelings_lol.count('Positive'), feelings_lol.count('Neutral'), feelings_lol.count('Negative'))
