import os
import io
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys
import re
#from langdetect import detect
import stop_words as s_w
import string
import numpy
import datetime

import predict as pred
import preprocess as ppc
#import nltk
#from nltk import tokenizerboi

#loads fewer files for more rapid testing
DEBUG_MODE = 0

BATCH_SIZE = 5
BATCH_LENGTH = 576000
LSTM_UNITS = 64
NUM_CLASSES = 2
BUFFER_SIZE = 600000
EPOCHS = 50
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

#list of all files being loaded

#get start time
programStartTime = datetime.datetime.now()

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
print("Importing Test Users")
#all tweets are stored in a single vector (string at the moment)
#let us consider storing each tweet in its own data feature. We can do an analysis on each tweet
depression_lengths = []
control_lengths = []
TWEETS_DEPRESSION = []
TWEETS_CONTROL    = []
FILE_STRINGS = []
FILE_STRINGS_CONTROL = []
count = 0
for files in FILE_NAMES_DEPRESSION:
    string_user = []
    for file in files:
        #print("Loaded tweets from depression: ", count, end='\r')
        tweet = ""
        all_lines = ""
        all_tweets = []
        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                if len(line) > 5:
                    if len(line) < 280:
                    	#pre process the lines as they come in
                        pps = ppc.pre_process_data(line)
                        if len(pps) > 1: 
                        	#pad lines
                            pps += [''] * (280 - len(pps))
                            TWEETS_DEPRESSION.append(pps)
                            count = count + 1
            if DEBUG_MODE == 1 and count > 10000:
                f.close()
                break   
            f.close()
count = 0
print('\n')
print("Importing Control Users")
for files in FILE_NAMES_CONTROL:
    string_user = []
    for file in files:
        #print("Loaded tweets from control: ", count, end='\r')
        tweet = ""
        all_lines = ""
        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                if len(line) > 5:
                    if len(line) < 280:
                        pps = ppc.pre_process_data(line)
                        if len(pps) > 1:
                            pps += [''] * (280 - len(pps))
                            TWEETS_CONTROL.append(pps)
                            count = count + 1
            if DEBUG_MODE == 1 and count > 10000:
                f.close()
                break
            f.close()
print('\n')

#count the number of features/data samples loaded
num_of_features_control = len(TWEETS_CONTROL)
num_of_features = len(TWEETS_DEPRESSION)
#create a list of labels for the samples
labels = numpy.array([1 for _ in range(num_of_features)])
labels_control = numpy.array([0 for _ in range(num_of_features_control)])
#concatenate the data sets
TWEETS = TWEETS_DEPRESSION + TWEETS_CONTROL

print("Test and Control Combined")
labels = numpy.concatenate([labels, labels_control])
print(len(TWEETS))
print(len(labels))

#create the data set
dataset = tf.data.Dataset.from_tensor_slices((TWEETS, labels))
print(dataset)

NUM_SAMPLES = num_of_features + num_of_features_control
print("Tweets Loaded: ", NUM_SAMPLES)

dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
#pred_dataset = pred.create_pred_set()

#create the dictionary that will be used for encoding
vocab_set = set()
for text_tensor, _ in dataset:
    some_tokens = text_tensor.numpy()
    vocab_set.update(some_tokens)
"""
for text_tensor, _ in pred_dataset:
    some_tokens = text_tensor.numpy()
    vocab_set.update(some_tokens)
"""
vocab_size = len(vocab_set)
vocab_set = sorted(vocab_set)

print("VOCAB SIZE: ", (vocab_size))
#create the encoder
encoder = tfds.features.text.TokenTextEncoder(vocab_set)

print("Encoding the Dataset")
#encode the data
def encode(text_tensor, label):
    string_text = b" ".join(text_tensor.numpy())
    #remove leading and ending whitespace
    string_text = string_text.strip()
    encoded_text = encoder.encode(string_text)
    return encoded_text, label

def encode_map_fn(text, label):
    return tf.py_function(encode, inp=(text, label), Tout=(tf.int64, tf.int64))
max_len_tensor = -1
dataset = dataset.map(encode_map_fn)
#pred_dataset = pred_dataset.map(encode_map_fn)
print("After Encoding")
print("Count -> data_set: ", count)
print("Batch Size: ", BATCH_SIZE)

TAKE_SIZE = NUM_SAMPLES//(10 * 2)

print("Take Size: ", TAKE_SIZE)

#batch the data
padded_shapes = ([280], [])
dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes, drop_remainder=True)
#create the seperated data sets
train_data = dataset.skip(TAKE_SIZE).shuffle(TAKE_SIZE)
pred_dataset = dataset.skip(TAKE_SIZE)
test_data = dataset.take(TAKE_SIZE)

count = 0
for elem in train_data:
    count = count + 1
print("Count -> Train_data:", count)
count = 0
for elem in test_data:
    count = count + 1
print("Count -> Test_data:", count)

"""
embedding_index = {}
in_embedding = io.open('embedding/embe_60.tsv', 'r', encoding='utf-8')
for line in in_embedding:
    values = line.split()
    word = values[0]
    coefs = numpy.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()

embedding_matrix = numpy.zeros((len(vocab_set) + 1, EMBEDDING_DIM))
i = 0
for word in vocab_set:
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    i = i + 1

embedding_layer = tf.keras.layers.Embedding(len(vocab_set) + 1,EMBEDDING_DIM,weights=[embedding_matrix],trainable=False)
"""

#create the directory that the model will be saved to
currentDT = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
o_dir = r"output_ST/" + str(currentDT)
os.mkdir(o_dir)

#checkpoint_path = o_dir + "/training/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = o_dir + "/training/cp.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

vocab_size += 1 #because we are adding 0 padding
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM))
model.add(tf.keras.layers.Conv1D(32, kernel_size=3, padding='valid', strides=1, activation='relu', input_shape=(BATCH_SIZE, 280, None))) #ideas: change this higher to around 128/256??
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.MaxPooling1D(pool_size=16))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(26, dropout=0.5)))
#model.add(tf.keras.layers.LSTM(64, input_shape=(None, None, None), dropout=0.5))
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
"""
"""
LSTM256 = tf.keras.Sequential()
LSTM256.add(tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM))
#LSTM256.add(tf.keras.layers.LSTM(128, input_shape=(BATCH_SIZE, 280, EMBEDDING_DIM), dropout=0.5))
LSTM256.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, input_shape=(BATCH_SIZE, 280, EMBEDDING_DIM), dropout=0.5)))
#LSTM256.add(tf.keras.layers.Dense(128, activation='relu'))
LSTM256.add(tf.keras.layers.Dense(64, activation='relu'))
LSTM256.add(tf.keras.layers.Dropout(0.5))
LSTM256.add(tf.keras.layers.Dense(16, activation='relu'))
LSTM256.add(tf.keras.layers.Dropout(0.5))
LSTM256.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model = LSTM256
"""
LSTM256 = tf.keras.Sequential()
LSTM256.add(tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM))
LSTM256.add(tf.keras.layers.LSTM(256, input_shape=(BATCH_SIZE, 280, EMBEDDING_DIM), dropout=0.5))
LSTM256.add(tf.keras.layers.Dense(64, activation='relu'))
LSTM256.add(tf.keras.layers.Dropout(0.5))
LSTM256.add(tf.keras.layers.Dense(16, activation='relu'))
LSTM256.add(tf.keras.layers.Dropout(0.5))
LSTM256.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model = LSTM256



print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, validation_data=test_data, epochs=EPOCHS, callbacks=[cp_callback, tensorboard_callback])#, batch_size=BATCH_SIZE)
os.mkdir(o_dir + "/model")
model.save(o_dir + "/model")

"""

predictions = model.predict(pred_dataset)
#print(predictions)
temp = pred_dataset
i = 0
for f,l in temp:
    for j in range(0,5):
        print(predictions[i+j])
    print(l)
    i = i + 1
"""

model.evaluate(pred_dataset)
#get the model's stats
test_loss, test_acc = model.evaluate(test_data)
print("Test Loss: {}", format(test_loss))
print("Test Accuracy: {}", format(test_acc))

programEndTime = datetime.datetime.now()

programRunTime = programEndTime - programStartTime
model_summary = model.summary()
summary_text = "Training final LSTM256 on 50 epochs. but it will be queues after the other slurm job."

#save the summary file and the trained embeddings

try:
    with open(o_dir + "/" + currentDT + ".txt", "w") as output:
        output.write("Accuracy: " + str(test_acc) + '\n')
        output.write("Loss: " + str(test_loss) + '\n')
        output.write("Batch Size: " + str(BATCH_SIZE) + '\n')
        output.write("Num data: " + str(len(labels)) + '\n')
        output.write("Epochs: " + str(EPOCHS) + '\n')
        output.write("Time Taken: " + str(programRunTime) + '\n')
        output.write("checkpoint_path=" + str(checkpoint_path) + '\n')
        output.write("vocab_size=" + str(vocab_size) + '\n')
        model.summary(print_fn=lambda x: output.write(x + '\n'))
        output.write(summary_text + '\n')
        output.close()
except Exception as e:
    print(e)

e = model.layers[0]
weights = e.get_weights()[0]
try:
    out_v = io.open(o_dir + "/" + 'vecs_'+ str(EMBEDDING_DIM) +'.tsv', 'w', encoding='utf-8')
    out_m = io.open(o_dir + "/" + 'meta_'+ str(EMBEDDING_DIM) +'.tsv', 'w', encoding='utf-8')

    for num, word in enumerate(encoder.tokens):
        vec = weights[num+1]
        out_m.write(word+"\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")

    out_m.close()
    out_v.close()
except Exception as e:
    print(e)
print("Finished")
