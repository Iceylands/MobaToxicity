import tweepy
import configparser
import pandas as pd

#read creds from config
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

#authenticate and instance the API
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#setting up keywords and limits
#no retweets for now
keywords_league = 'League of Legends -RT'
keywords_dota = 'dota 2 -RT'
limit = 300

League_tweets = tweepy.Cursor(api.search_tweets, q=keywords_league, lang='en', count = 100,
                              tweet_mode= 'extended').items(limit)
Dota_tweets = tweepy.Cursor(api.search_tweets, q=keywords_dota, lang='en', count = 100,
                              tweet_mode= 'extended').items(limit)

columns = ['Tweet']
data_lol = []
data_dota = []

for tweet in League_tweets:
    data_lol.append([tweet.full_text])

for tweet in Dota_tweets:
    data_dota.append([tweet.full_text])

df_lol = pd.DataFrame(data_lol, columns = columns)
df_dota = pd.DataFrame(data_dota, columns = columns)

df_lol.to_csv('tweets_lol.csv')
df_dota.to_csv('tweets_dota.csv')

#for tweet in public_tweets:
#    print(tweet.text)
