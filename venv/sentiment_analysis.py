from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

tweet = "@BobassdreShakarami I guess league is one of those games you love to hate haha @LeagueOfLegends @Tylerxd :V"

#preprocess tweet
tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    elif word.startswith('http'):
        word = 'http'
    tweet_words.append(word)

tweet_proc = " ".join(tweet_words)

print(tweet_proc)

#loading the model and tokenizer

roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative','Neutral','Positive']

#Analysis

encoded_tweet = tokenizer(tweet_proc, return_tensors = 'pt')
print(encoded_tweet)

output = model(**encoded_tweet)

print(output)

scores =output[0][0].detach().numpy()
scores =softmax(scores)

for i in range(len(scores)):
    l = labels[i]
    s = scores[i]

    print(l,s)