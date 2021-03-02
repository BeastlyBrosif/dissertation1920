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

import stop_words as s_w
import string
import numpy
import datetime

import preprocess as ppc
import predict as pred
#THIS CODE IS OLD AND NOT UP TO DATE WITH DATASET_BUILDER.PY

#import nltk
#from nltk import tokenizerboi

BATCH_SIZE = 12
BATCH_LENGTH = 576000
LSTM_UNITS = 64
NUM_CLASSES = 2
BUFFER_SIZE = 400
EMBEDDING_DIM = 60
EPOCHS = 300

NUM_DIMENSIONS = 300
MAX_SEQUENCE_LENGTH = 180
MAX_FILES_PER_USER  = 3200
#This is how many tweets the user has. The program will find the max number of Tweets that any user has
#and set this variable to that. Then, any users who don't have enough tweets, the values will be padded with zero
TENSOR_WIDTH = -1
MAX_TENSOR_LENGTH = MAX_SEQUENCE_LENGTH * MAX_FILES_PER_USER

def labeler(exapmle, index):
	return example, tf.cast(index, tf.int64)

#list of all files being loaded


#TF1 stop list

def old_pre_process_data(line):
	#convert the string to lower case
        detected_lang = ""
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
    if len(string_user) < MAX_TENSOR_LENGTH:
        string_user += [''] * (MAX_TENSOR_LENGTH - len(string_user))
    #print(len(string_user))
    count = count + 1
    FILE_STRINGS.append(string_user)
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
    if len(string_user) < MAX_TENSOR_LENGTH:
        string_user += [''] * (MAX_TENSOR_LENGTH - len(string_user))
    #print(len(string_user))
    count = count + 1
    FILE_STRINGS_CONTROL.append(string_user)

num_of_features_control = len(FILE_STRINGS_CONTROL)
num_of_features = len(FILE_STRINGS)
labels = numpy.array([1 for _ in range(num_of_features)])
labels_control = numpy.array([0 for _ in range(num_of_features_control)])
FILE_STRINGS = FILE_STRINGS + FILE_STRINGS_CONTROL

print("Test and Control Combined")
labels = numpy.concatenate([labels, labels_control])
print("Labels: ", labels)
dataset = tf.data.Dataset.from_tensor_slices((FILE_STRINGS, labels))

NUM_USERS = num_of_features + num_of_features_control
print("Users Loaded: ", NUM_USERS)

dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
pred_dataset = pred.create_pred_set()

vocab_set = set()
for text_tensor, _ in dataset:
    some_tokens = text_tensor.numpy()
    vocab_set.update(some_tokens)
for text_tensor, _ in pred_dataset:
    some_tokens = text_tensor.numpy()
    vocab_set.update(some_tokens)
vocab_size = len(vocab_set)
vocab_set = sorted(vocab_set)

print("VOCAB SIZE: ", (vocab_size))

encoder = tfds.features.text.TokenTextEncoder(vocab_set)

def encode(text_tensor, label):
    string_text = b" ".join(text_tensor.numpy())
    #remove leading and ending whitespace
    string_text = string_text.strip()
    encoded_text = encoder.encode(string_text)
    return encoded_text, label

def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

max_len_tensor = -1
dataset = dataset.map(encode_map_fn)
pred_dataset = pred_dataset.map(encode_map_fn)

count = 0
for elem in dataset:
    count = count + 1
    feature, label = elem
    #print(feature.shape, "(", label, ")")

print("Count -> data_set: ", count)
print("Batch Size: ", BATCH_SIZE)

TAKE_SIZE = NUM_USERS//10 * 1

print("Take Size: ", TAKE_SIZE)

train_data = dataset.skip(TAKE_SIZE)
#train_data = train_data.padded_batch(BATCH_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=(([None],[])))

test_data = dataset.take(TAKE_SIZE)
#test_data = test_data.padded_batch(BATCH_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=(([None],[])))

count = 0
for elem in train_data:
    feature, label = elem
    #print(feature.shape, "(", label, ")")
    count = count + 1

print("Count -> Train_data:", count)


count = 0
for elem in test_data:
    count = count + 1
print("Count -> Test_data:", count)
checkpoint_path = "embedding/training/20200424-164408_d60/cp.ckpt"
#checkpoint_path = "embedding/training/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_d"+str(EMBEDDING_DIM)+"/cp.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

vocab_size += 1 #because we are adding 0 padding

#add dropout to the model, after the LSTM layer

#e = tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_mat], input_length???, trainable=False)

e = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)
_
model = tf.keras.Sequential()
model.add(e)
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='softmax'))
model.add(tf.keras.layers.Dense(1))

print(model.summary())
model.load_weights(checkpoint_path)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_data, validation_data=test_data, epochs=EPOCHS, callbacks=[cp_callback, tensorboard_callback], verbose=1)#, batch_size=BATCH_SIZE)

#get the model's stats
test_loss, test_acc = model.evaluate(test_data)
print("Test Loss: {}", format(test_loss))
print("Test Accuracy: {}", format(test_acc))

currentDT = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
with open('embedding/' + str(EMBEDDING_DIM) + "_EMBEDDINGS.txt", "w") as output:
    output.write("Accuracy: " + str(test_acc) + '\n')
    output.write("Loss: " + str(test_loss) + '\n')
    output.write("Batch Size: " + str(BATCH_SIZE) + '\n')
    output.write("Num data: " + str(len(labels)) + '\n')
    output.write("Epochs: " + str(EPOCHS) + '\n')
    output.write("checkpoint_path=" + str(checkpoint_path) + '\n')
    model.summary(print_fn=lambda x: output.write(x + '\n'))
    output.close()

"""
out_v = io.open('embedding/vecs_60.tsv', 'w', encoding='utf-8')
out_m = io.open('embedding/meta_'+str(EMBEDDING_DIM)+'.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.tokens):
    vec = weights[num+1]
    out_m.write(word+"\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()
"""

e = model.layers[0]
weights = e.get_weights()[0]
try:
    #out_v  = io.open('embedding/vecs_' + str(EMBEDDING_DIM) + '.tsv', 'w', encoding='utf-8')
    #out_m  = io.open('embedding/meta_' + str(EMBEDDING_DIM) + '.tsv', 'w', encoding='utf-8')
    out_60 = io.open('embedding/embe_' + str(EMBEDDING_DIM) + '.tsv', 'w', encoding='utf-8')
    for num, word in enumerate(encoder.tokens):
        vec = weights[num+1]
        out_60.write(word + '\t' + '\t'.join([str(x) for x in vec]) + "\n")
        #out_m.write(word+"\n")
        #out_v.write('\t'.join([str(x) for x in vec]) + "\n")

    out_60.close()
    #out_m.close()
    #out_v.close()
except Exception as ex:
    print(ex)

print("END")


