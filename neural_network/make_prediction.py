import os
import io
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
import nltk
from nltk.tokenize import word_tokenize
import sys
import re
from langdetect import detect
import stop_words as s_w
import string
import numpy
import datetime

import preprocess as ppc

MAX_TENSOR_LENGTH = 576000 #could be lower??
vocab_size = 0
LENGTHS_DEPRESSION = []
LENGTHS_CONTROL    = []


def get_strings(FILE_NAMES, b_dep):
    count = 0
    FILE_STRINGS = []
    for files in FILE_NAMES:
        string_user = []
        for file in files:
            tweet = ""
            all_lines = ""
            with open(file, 'r', encoding="utf-8") as f:
                for line in f:
                    all_lines += line
                f.close()
            pre_processed_tokens = ppc.pre_process_data(all_lines)
            for ppt in pre_processed_tokens:
                string_user.append(ppt)
            #string_user.append(pre_processed_tokens)
        #string_user = ''.join(string_user)
        if len(string_user) < 1000:
            print("User Omitted: Too Few Records")
            continue
        if len(string_user) < MAX_TENSOR_LENGTH:
            if b_dep == 1:
                LENGTHS_DEPRESSION.append(len(string_user))
            else:
                LENGTHS_CONTROL.append(len(string_user))
            string_user += [''] * (MAX_TENSOR_LENGTH - len(string_user))
        #print(len(string_user))
        count = count + 1
        FILE_STRINGS.append(string_user)
        if count > 999:
            break
    return FILE_STRINGS


def create_pred_set():
    FILE_NAMES_DEPRESSION = []
    FILE_NAMES_CONTROL    = []
    DEPRESSION_DIR = "../twurl/twitter_data/predict/depression/"
    CONTROL_DIR = "../twurl/twitter_data/predict/control/"
    for subdir in next(os.walk(DEPRESSION_DIR))[1]:
            SUB_DIR = subdir
            user = []
            for file in os.listdir(DEPRESSION_DIR + SUB_DIR):
                    if file.endswith(".txt"):
                            file_name = os.path.join(DEPRESSION_DIR + "/" + SUB_DIR, file)
                            user.append(file_name)
            FILE_NAMES_DEPRESSION.append(user)
    for subdir in next(os.walk(CONTROL_DIR))[1]:
        SUB_DIR = subdir
        user = []
        for file in os.listdir(CONTROL_DIR + SUB_DIR):
            if file.endswith(".txt"):
                file_name = os.path.join(CONTROL_DIR + "/" + SUB_DIR, file)
                user.append(file_name)
        FILE_NAMES_CONTROL.append(user)
    FILE_STRINGS_DEPRESSION = []
    FILE_STRINGS_CONTROL    = []
    FILE_STRINGS_DEPRESSION = get_strings(FILE_NAMES_DEPRESSION, 1)
    FILE_STRINGS_CONTROL    = get_strings(FILE_NAMES_CONTROL, 0)
    #print(len(FILE_STRINGS_DEPRESSION))
    #print(len(FILE_STRINGS_CONTROL))
    num_features_control = len(FILE_STRINGS_CONTROL)
    num_features_depress = len(FILE_STRINGS_DEPRESSION)
    labels = numpy.array([1 for _ in range(num_features_depress)])
    labels_control = numpy.array([1 for _ in range(num_features_control)])
    FILE_STRINGS = FILE_STRINGS_DEPRESSION + FILE_STRINGS_CONTROL
    #print("Dep and Control Combined")
    labels = numpy.concatenate([labels, labels_control])
    dataset = tf.data.Dataset.from_tensor_slices((FILE_STRINGS, labels))
    NUM_USERS = num_features_control + num_features_depress
    print("Prediction Dataset: ", dataset)
    return dataset



checkpoint_path = "output/2020_04_09_12-21-59/training/cp.ckpt"
checkpoint_dir  = os.path.dirname(checkpoint_path)


print("Using model ", checkpoint_path, " to make predictions")


BATCH_SIZE = 5
BATCH_LENGTH = 576000
LSTM_UNITS = 64
NUM_CLASSES = 2
BUFFER_SIZE = 100
EPOCHS = 10
EMBEDDING_DIM = 8

NUM_DIMENSIONS = 300
MAX_SEQUENCE_LENGTH = 180
MAX_FILES_PER_USER  = 3200
#This is how many tweets the user has. The program will find the max number of Tweets that any user has
#and set this variable to that. Then, any users who don't have enough tweets, the values will be padded with zero
TENSOR_WIDTH = -1
MAX_TENSOR_LENGTH = MAX_SEQUENCE_LENGTH * MAX_FILES_PER_USER

def labeler(exapmle, index):
	return example, tf.cast(index, tf.int64)

def old_pre_process_data(line):
	#convert the string to lower case
        detected_lang = ""
        original_line = line
        line = line.lower()
        #remove responding to tweets
        line = re.sub(r"http://t.co\S+", "", line)
        line = re.sub(r"https://t.co\S+", "", line)
        line = re.sub(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)", "HANDLE", line)
	#replace URLS in string with 'URL'
        line = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", 'URL', line)
        line = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", 'URL', line)
	#replace countries, cities, and names.
        line = re.sub(r'[^\x00-\x7f]',r'', line) 
        line = line.translate(str.maketrans('','',string.punctuation + "’"))
        line = word_tokenize(line)
        original_line = word_tokenize(original_line)
        for word in line:
            if word in s_w.countries:# and 1 == 0:
                word.replace(word, 'COUNTRY')
            if word in s_w.cities:# and 1 == 0:
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

programStartTime = datetime.datetime.now()

longest_record = 0

FILE_NAMES_DEPRESSION = []
FILE_NAMES_CONTROL    = []
DEPRESSION_DIR = "../twurl/twitter_data/depression/"
CONTROL_DIR = "../twurl/twitter_data/control/"
#collect all the subdirectories (users) in the directory 'DEPRESSION_DIR'
for subdir in next(os.walk(DEPRESSION_DIR))[1]:
	SUB_DIR = subdir
	user = []
	for file in os.listdir(DEPRESSION_DIR + SUB_DIR):
		if file.endswith(".txt"):
			file_name = os.path.join(DEPRESSION_DIR + "/" + SUB_DIR, file)
			user.append(file_name)
	FILE_NAMES_DEPRESSION.append(user)
#collect all the subdirectories (users) in the directory 'CONTROL_DIR'
for subdir in next(os.walk(CONTROL_DIR))[1]:
    SUB_DIR = subdir
    user = []
    for file in os.listdir(CONTROL_DIR + SUB_DIR):
        if file.endswith(".txt"):
            file_name = os.path.join(CONTROL_DIR + "/" + SUB_DIR, file)
            user.append(file_name)
    FILE_NAMES_CONTROL.append(user)

#now we turn the file_names_X into long vectors of text
#max vector length 3200 * MAX TWITTER CHARACTERS (180)
print("Importing Test Users")
#all tweets are stored in a single vector (string at the moment)
depression_lengths = []
control_lengths = []
FILE_STRINGS = []
FILE_STRINGS_CONTROL = []
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
        pre_processed_tokens = ppc.pre_process_data(all_lines)
        #print(pre_processed_tokens) #debugging line
        for ppt in pre_processed_tokens:
            string_user.append(ppt)
        #string_user.append(pre_processed_tokens)
    #string_user = ''.join(string_user)
    if len(string_user) < 1000:
        print("User Omitted: Too Few Records")
        continue
    if len(string_user) < MAX_TENSOR_LENGTH:
        depression_lengths.append(len(string_user))
        string_user += [''] * (MAX_TENSOR_LENGTH - len(string_user))
    #print(len(string_user))
    count = count + 1
    FILE_STRINGS.append(string_user)
    if count > 999:
        break
count = 0
print("Importing Control Users")
for files in FILE_NAMES_CONTROL:
    string_user = []
    for file in files:
        tweet = ""
        all_lines = ""
        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                all_lines += line
            f.close()
        pre_processed_tokens = ppc.pre_process_data(all_lines)
        #print(pre_processed_tokens) #debugging line
        for ppt in pre_processed_tokens:
            string_user.append(ppt)
        #string_user.append(pre_processed_tokens)
    #string_user = ''.join(string_user)
    if len(string_user) < 1000:
        print("User Omitted: Too Few Records")
        continue
    if len(string_user) < MAX_TENSOR_LENGTH:
        control_lengths.append(len(string_user))
        string_user += [''] * (MAX_TENSOR_LENGTH - len(string_user))
    #print(len(string_user))
    count = count + 1
    FILE_STRINGS_CONTROL.append(string_user)
    if count > 999:
       break

#print(control_lengths)
#print(depression_lengths)
#exit()
num_of_features_control = len(FILE_STRINGS_CONTROL)
num_of_features = len(FILE_STRINGS)
labels = numpy.array([1 for _ in range(num_of_features)])
labels_control = numpy.array([0 for _ in range(num_of_features_control)])
FILE_STRINGS = FILE_STRINGS + FILE_STRINGS_CONTROL


#FILE_STRINGS.append(FILE_STRINGS_CONTROL))
print("Test and Control Combined")
labels = numpy.concatenate([labels, labels_control])
print("Labels: ", labels)
dataset = tf.data.Dataset.from_tensor_slices((FILE_STRINGS, labels))
NUM_USERS = num_of_features + num_of_features_control
print("Users Loaded: ", NUM_USERS)

dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
pred_dataset = create_pred_set()


example = pred_dataset.take(5)
for features, label in example:
    print(features)


vocab_set = set()
for text_tensor, _ in dataset:
    some_tokens = text_tensor.numpy()
    vocab_set.update(some_tokens)
print(len(vocab_set))
for text_tensor, _ in pred_dataset:
    some_tokens = text_tensor.numpy()
    vocab_set.update(some_tokens)
print(len(vocab_set))
vocab_size = len(vocab_set)
vocab_set = sorted(vocab_set)

print("VOCAB SIZE: ", (vocab_size))

vocab_size = len(vocab_set)
vocab_size = vocab_size + 1
encoder = tfds.features.text.TokenTextEncoder(vocab_set)

print("Encoding the Dataset")
def encode(text_tensor, label):
    string_text = b" ".join(text_tensor.numpy())
    #remove leading and ending whitespace
    string_text = string_text.strip()
    encoded_text = encoder.encode(string_text)
    output = (encoded_text, label)
    return encoded_text, label

def encode_map_fn(text, label):
    #return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))
    return tf.py_function(encode, inp=(text, label), Tout=(tf.int64, tf.int64))

max_len_tensor = -1
pred_dataset = pred_dataset.map(encode_map_fn)
print("After Encoding")

example = pred_dataset.take(5)
for elem in example:
    print(elem)

max_len_tensor = -1
dataset = dataset.map(encode_map_fn)
print("After Encoding")

example = dataset.take(2)
for elem in example:
    feature, label = elem
    print(elem)

padded_shapes = (tf.TensorShape([None]), tf.TensorShape([]))
pred_dataset = pred_dataset.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 8))
#model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24,input_shape=(None, None, None),  dropout=0.5)))
model.add(tf.keras.layers.LSTM(24, input_shape=(None,None,None), dropout=0.5))
model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(8, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(8, activation='softmax')) #note- why 8. This means we 'abstract' 8 features from the LSTM layer. But why?
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #formerly softmax activation

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
loss, acc = model.evaluate(pred_dataset, verbose=2)
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(pred_dataset, verbose=2)
model.predict(pred_dataset)

