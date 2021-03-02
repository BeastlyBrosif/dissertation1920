#from tweepy.streaming import StreamListener

#from tweepy import Stream
import tweepy
import json
import os
from tweepy import OAuthHandler

import twitter_credentials

#authorising the Twitter API using the credentials of the other file
auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

user = "BiggECheese2"
user_number = str(1)

def get_user_tweets(destination_dir, dir_number, user_handle):
    #create the directory if it doesn't already exist
    #example destination_dir = control or depression
    destination_dir = "twitter_data/predict/"  + str(destination_dir) + "/" + str(dir_number).zfill(5) + "/"
    dir = os.path.join(destination_dir)
    print(dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    #Keep some variables for the statistics of each user, these will not be used, necessarily.
    maxFav = 0
    avgFav = 0
    totalFav = 0
    count = 0;
    #maxTweetText = ""
    #This for loop gets all non-retweeted
    for tweet in tweepy.Cursor(api.user_timeline,id=user_handle, tweet_mode='extended', wait_on_rate_limit=True).items():
        #print(tweet)
        #print(tweet.text)
        #print(tweet.favorite_count)
        if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
            count = count + 1
            totalFav = totalFav + tweet.favorite_count
            tweetfilename = dir + str(count).zfill(5) + ".txt"
            with open(tweetfilename, 'a', encoding="utf-8") as file:
                #print(tweet.text)
                file.write(tweet.full_text)
                file.close()
            # with open(filename, 'a') as tf:
            # 	tf.write(json.dumps(tweet._json))
            # 	tf.write(',')
            #  tf.ciose()
            if count % 100 == 0:
                print("Loaded %d Tweets." % count)
            if tweet.favorite_count > maxFav:
                maxFav = tweet.favorite_count
                maxTweetText = tweet.full_text
    return maxFav, avgFav, totalFav, count
