COMP4027 By Nimrod Wynne
neural_network > contains python scripts for training nn designs
twurl > contains scripts that were used to acquire the data from twitter
##########################################################################
neural_network:
dataset_builder.py: creates the 'whole user' problem set and trains a neural network on the model defined in the problem
single_user.py: create the 'individual tweet' problem set and trains an nn on the data
preprocess.py: contains the pre-processing algorithm that is used when data is loaded from the data set
count_freqs.py: either A. creates vocab files for both control and depression groups or B. shows occurrences of words in both the depression and control set to be compared.
stop_words.py: contains a list of stop words that were used in the pre-processing algorithm. but these were locationa and names.
predict.py: loads the prediction group from the Twitter data and uses a model to get the prediction values for the 'whole user' problem. 
embedding_training.py: a script used to train word embeddings, however, what was produced was not used in the nn models.
##########################################################################
twurl:
control_user_get_tweets.py: a set of functions used to gather tweets from a user for the control group
get_control_set.py: scans the Twitter feed and selects from it users with more than 1,000 posted Tweets
search_tweets.py: retrieves users from the Twitter feed who have posted a Tweet containing the keywords "i have been diagnosed with depression"
tweepy_steamer.py: not used to collect data but to prototype the get_control_set.py
user_get_tweets.py: downloads tweets from a user who has announced their depression diagnosis. 
##########################################################################
Info on running the training program:
The python files could not be compiled into an exe. They can be run from a python environment containing the dependencies found in requirements.txt. 