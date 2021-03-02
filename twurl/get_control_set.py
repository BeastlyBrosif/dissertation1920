#from tweepy.streaming import StreamListener

#from tweepy import Stream
import tweepy
import json
import os
from tweepy import OAuthHandler

import twitter_credentials
import user_get_tweets as ugt

auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)
count = 0

#keeps track of 'unsure' inputs
ids = set()
num_of_users = 0;
dirs = 0
#for _,dirnames,_ in os.walk("twitter_data/control"):
for _,dirnames,_ in os.walk("twitter_data/predict/control"):
    dirs += len(dirnames)
    print("Directories: ", dirs)

num_of_users = dirs
with open("control_group_pred.txt", 'r+', encoding='utf-8') as file:
    for line in file:
        curr_user = line
        print(num_of_users, line)
        try: 
            maxFav, avgFav, totalFav, totalCount = ugt.get_user_tweets(destination_dir="control", dir_number=num_of_users, user_handle=line)
        except tweepy.TweepError:
            print("Tweepy Error!")
            pass
        num_of_users = num_of_users + 1
    file.close()

"""
for tweet in tweepy.Cursor(api.search, q="i diagnosed with depression", tweet_mode="extended", lang='en').items(50000):
	in_key = ''
	text = tweet.full_text
	if not tweet.retweeted and 'RT @' not in text:
		print(text)
		print(tweet.user.screen_name)
		print("------------------------------------------------------")
		in_key = input('Acceptable (y/n/u)?')
		b_newUser = True
		if in_key == 'y':
			with open("user_list_control.txt", 'r+', encoding='utf-8') as file:
				for line in file:
					if tweet.user.screen_name == line:
						print("This user has already been addeed to the file!")
						b_newUser = False
				if b_newUser == True:
					file.write(tweet.user.screen_name) # this is used so we know which users have been selected
					file.write("\n")
					file.close()
					#now download this users tweet
					maxFav, avgFav, totalFav, totalCount = ugt.get_user_tweets(destination_dir="control", dir_number=num_of_users, user_handle=tweet.user.screen_name)
					num_of_users = num_of_users + 1
		elif in_key == 'u':
			ids.add(tweet)
		count = count + 1

print("Count: %d" % (count))
"""
print("END")
