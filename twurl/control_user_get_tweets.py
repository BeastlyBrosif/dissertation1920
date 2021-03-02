import sys
import smtplib, ssl

#from tweepy import Stream
import tweepy
import json
import os
from tweepy import OAuthHandler
import time
from datetime import datetime
import twitter_credentials

#authorising the Twitter API using the credentials of the other file
auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

num_users_to_collect = sys.argv[1]
collected_users = 0

def add_user_to_list(user_name):
    global collected_users
    global num_users_to_collect
    with open("control_group.txt", "a") as controlfile:
        controlfile.write(user_name+"\n")
        controlfile.close()
        collected_users = collected_users + 1
    if collected_users > int(num_users_to_collect) or collected_users == int(num_users_to_collect):
        return 1

def get_num_tweets_user(user_name):
    count = 0
    backoff = 1
    try:
        for tweet in tweepy.Cursor(api.user_timeline, id=user_name, tweet_mode='extended', wait_on_rate_limit=True).items():
            if(not tweet.retweeted) and ('RT @' not in tweet.full_text):
                count = count + 1
                if count % 100 == 0:
                    print("Loaded ", count)
                    time.sleep(2)
    except tweepy.TweepError as e:
        print(e.reason)
        time.sleep(60*backoff)
        backoff = backoff + 1
    return count

class MyStreamListener(tweepy.StreamListener):
    global num_users_to_collect
    global collected_users
    def on_status(self, status):
        print(status.user.screen_name, "---", status.text)
        time.sleep(1)
        user_count = get_num_tweets_user(status.user.screen_name)
        if user_count > 1000:
            add_user_to_list(status.user.screen_name)
            print(status.user.screen_name + " has been added to control_group.txt")
    def on_error(self, status_code):
        if status_code == 503:
            return 0

##################################################################
#function returns an array of users that will be batch downloaded#
def get_users_control_group():
    global num_users_to_collect, collected_users
    backoff = 1
    mystreamlistener = MyStreamListener()
    mystream = tweepy.Stream(auth=api.auth, listener=mystreamlistener)
    while collected_users < int(num_users_to_collect):
        try:
            mystream.filter(track=["I"])
        except:
            print("An Exception occured. Waiting...")
            time.sleep(60*backoff)
            backoff = backoff + 1
            if backoff > 7:
                exit()
    #scan tweets
    #get user
    #check num of tweets
    #add username to list if num tweets > threshold

def get_user_control_tweets(destination_dir, dir_number, user_handle):
    #create the directory if it doesn't already exist
    #example destination_dir = control or depression
    destination_dir = "twitter_data/predict/"  + str(destination_dir) + "/" + str(dir_number).zfill(5) + "/"
    dir = os.path.join(destination_dir)
    print("dir: " + dir)
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
            if tweet.favorite_count > maxFav:
                maxFav = tweet.favorite_count
                maxTweetText = tweet.full_text
            if count % 100 == 0:
                print("Loaded %d Tweets." % count)
                time.sleep(2)
    return maxFav, avgFav, totalFav, count

get_users_control_group()

