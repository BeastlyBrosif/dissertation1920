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
import predict as pred


BATCH_SIZE = 5
BATCH_LENGTH = 576000
BUFFER_SIZE = 400
LSTM_UNITS = 64
NUM_CLASSES = 2
EPOCHS = 10
EMBEDDING_DIM = 60

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




#for statistics - check how long it took to run the program
programStartTime = datetime.datetime.now()

longest_record = 0
#how many samples did the program ommit
data_samples_ommited = 0
#the minimum length a sample should be if it is to be accepted into the training set
MIN_SAMPLE_LENGTH = 1000


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
    if len(string_user) < MIN_SAMPLE_LENGTH:
        data_samples_ommited = data_samples_ommited + 1
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
    if len(string_user) < MIN_SAMPLE_LENGTH:
        data_samples_ommited = data_samples_ommited + 1
        continue
    if len(string_user) < MAX_TENSOR_LENGTH:
        control_lengths.append(len(string_user))
        string_user += [''] * (MAX_TENSOR_LENGTH - len(string_user))
    #print(len(string_user))
    count = count + 1
    FILE_STRINGS_CONTROL.append(string_user)
    if count > 999:
       break

print(str(data_samples_ommited) + " were ommited. Reason: Length less than " + str(MIN_SAMPLE_LENGTH))



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
pred_dataset = pred.create_pred_set()


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
#vocab_size = vocab_size + 1
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



#example = dataset.take(5)
#for features, label in example:
    #print(features , label)
"""
count = 0
example = dataset.take(5)
for elem in example:
    count = count + 1
    feature, label = elem
    print(feature.shape, "(", label, ")")
"""
print("Count -> data_set: ", count)
print("Batch Size: ", BATCH_SIZE)

TAKE_SIZE = NUM_USERS//10 * 2

print("Take Size: ", TAKE_SIZE)

#padded_shapes = ([576000, None], [])
padded_shapes = (tf.TensorShape([None]), tf.TensorShape([])) #maybe change tf.TensorShape([576000]) to tf.TensorShape([None])
#datatset = dataset.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)

train_data = dataset.skip(TAKE_SIZE).shuffle(TAKE_SIZE)
#train_data = train_data.batch(BATCH_SIZE)
#train_data = train_data.padded_batch(BATCH_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)

"""
examples = train_data.take(5)
for elem in examples:
    feature, label = elem
    #print(feature, label)
"""
test_data = dataset.take(TAKE_SIZE)
#test_data = dataset.batch(BATCH_SIZE)
#test_data = test_data.padded_batch(BATCH_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)
"""
count = 0
for elem in train_data:
    feature, label = elem
    #print(feature, label)
    count = count + 1
"""
#print("Count -> Train_data:", count)

"""
count = 0
for elem in test_data:
    count = count + 1
"""
#print("Count -> Test_data:", count)

print(train_data)
print(test_data)

currentDT = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

o_dir = r"output/" + str(currentDT)
os.mkdir(o_dir)

#checkpoint_path = o_dir + "/training/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint+path = o_dir + "/training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

vocab_size += 1 #because we are adding 0 padding


embedding_index = {}
in_v = io.open('embedding/vecs_60.tsv', 'r', encoding='utf-8')
in_m = io.open('embedding/meta_60.tsv', 'r', encoding='utf-8')
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


model = tf.keras.Sequential()
#model.add(tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM))
model.add(embedding_layer)
model.add(tf.keras.layers.LSTM(64, dropout=0.5, return_sequences=True)) #input_shape=(None, None, None)
model.add(tf.keras.layers.LSTM(16, dropout=0.5))
#model.add(tf.keras.layers.Dense(8, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(16, activation='relu')) #note- why 8. This means we 'abstract' 8 features from the LSTM layer. But why?
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #formerly softmax activation
print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=test_data, epochs=EPOCHS, callbacks=[cp_callback, tensorboard_callback])#, batch_size=BATCH_SIZE)

model.evaluate(pred_dataset, verbose=2)

model_dir = o_dir + "/model"
os.mkdir(model_dir)
model.save(model_dir)
#get the model's stats
test_loss, test_acc = model.evaluate(test_data)
print("Test Loss: {}", format(test_loss))
print("Test Accuracy: {}", format(test_acc))

predictions = model.predict(predict_dataset)
print(predictions)
programEndTime = datetime.datetime.now()

programRunTime = programEndTime - programStartTime

#currentDT = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
model_summary = model.summary()
summary_text = "okay, complete overhaul. Now using two LSTM layers stacked. The first one returns the sequences. The embedding layer is one that was pre trained on the dataset. The pre processing algorithm doesnt remove stopwords or loc+names anymore but now does remove digits"

#perform PREDICTION now

try:
    with open(o_dir + "/" + currentDT + ".txt", "w") as output:
        output.write("Accuracy: " + str(test_acc) + '\n')
        output.write("Loss: " + str(test_loss) + '\n')
        output.write("Batch Size: " + str(BATCH_SIZE) + '\n')
        output.write("Num data: " + str(len(labels)) + '\n')
        output.write("Epochs: " + str(EPOCHS) + '\n')
        output.write("Time Taken: " + str(programRunTime) + '\n')
        output.write("checkpoint_path=" + str(checkpoint_path) + '\n')
        model.summary(print_fn=lambda x: output.write(x + '\n'))
        output.write(summary_text + '\n')
        output.close()
except Exception as e:
    print(e)
"""
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
"""
print("Finished")
