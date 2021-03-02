import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
import os
import re
import stop_words as s_w
import string

#import nltk
#from nltk import tokenizerboi

BATCH_SIZE = 24
LSTM_UNITS = 64
NUM_CLASSES = 2
BUFFER_SIZE = 10000

NUM_DIMENSIONS = 300
MAX_SEQUENCE_LENGTH = 180
MAX_FILES_PER_USER  = 3200
#This is how many tweets the user has. The program will find the max number of Tweets that any user has
#and set this variable to that. Then, any users who don't have enough tweets, the values will be padded with zero
TENSOR_WIDTH = -1


def labeler(exapmle, index):
	return example, tf.cast(index, tf.int64)

#list of all files being loaded


#TF1 stop list

def pre_process_data(line):
	#convert the string to lower case
        line = line.lower()
        #remove responding to tweets
        line = re.sub(r"http://t.co\S+", "", line)
        line = re.sub(r"https://t.co\S+", "", line)
	#replace URLS in string with 'URL'
        line = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", 'URL', line)
        line = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", 'URL', line)
	#replace countries, cities, and names.
        line = line.translate(str.maketrans('','',string.punctuation))
        line = word_tokenize(line)
        for word in line:
            if word in s_w.countries:
                word.replace(word, 'COUNTRY')
            if word in s_w.cities:
                word.replace(word, 'LOCATION')
            letter = word[0]
            #convert first char to an int
            num = ord(letter)
            if num >= 97 and num <= 100:
                if word in s_w.names_AD:
                    word.replace(word, 'NAME')
            elif num >= 101 and num <= 103:
                if word in s_w.names_EG:
                    word.replace(word, 'NAME')
            elif num >= 104 and num <= 108:
                if word in s_w.names_HL:
                    word.replace(word, 'NAME')
            elif num >= 109 and num <= 115:
                if word in s_w.names_MS:
                    word.replace(word, 'NAME')
            elif num >= 116 and num <= 122:
                if word in s_w.names_TZ:
                    word.replace(word, 'NAME')
        return line



#TEST_DIR  = ""~/dissertation/dissertation1920/twurl/twitter_data/depression/


FILE_NAMES_DEPRESSION = []
FILE_NAMES_CONTROL    = ["~/dissertation/dissertation1920/twurl/twitter_data/control/"]
DEPRESSION_DIR = "../twurl/twitter_data/depression/"

#collect all the subdirectories (users) in the directory 'DEPRESSION_DIR'
for subdir in next(os.walk(DEPRESSION_DIR))[1]:
	SUB_DIR = subdir
	user = []
	print(subdir)
	for file in os.listdir(DEPRESSION_DIR + SUB_DIR):
		if file.endswith(".txt"):
			file_name = os.path.join(DEPRESSION_DIR + "/" + SUB_DIR, file)
			user.append(file_name)
	FILE_NAMES_DEPRESSION.append(user)

#now we turn the file_names_X into long vectors of text
#max vector length 3200 * MAX TWITTER CHARACTERS (180)

#all tweets are stored in a single vector (string at the moment)
FILE_STRINGS = []
count = 0
for files in FILE_NAMES_DEPRESSION:
    string_user = []
    for file in files:
        tweet = ""
        all_lines = ""
        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                all_lines += line
            f.close()
        pre_processed_tokens = pre_process_data(all_lines)
        print(pre_processed_tokens)
        string_user.append(pre_processed_tokens)
    count = count + 1
    if count > 0:
        break
    FILE_STRINGS.append(string_user)	
"""

for i, file_name in enumerate(FILE_NAMES):
	lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
	labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
	labeled_data_sets.append(labeled_dataset)

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
  
all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
for ex in all_labeled_data.take(5):
  print(ex)

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
"""
