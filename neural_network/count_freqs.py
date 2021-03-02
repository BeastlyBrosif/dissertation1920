import matplotlib.pyplot as plt
import os
import io
import matplotlib.pyplot as plt
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
#import tensorflow_datasets as tfds
import nltk
from nltk.tokenize import word_tokenize
import sys
import re
from langdetect import detect
import stop_words as s_w
import string
import numpy
import datetime

import argparse

import preprocess as ppc
#import predict as pred
from collections import defaultdict
from nltk.tokenize import word_tokenize

import preprocess as ppc

DEBUG_MODE = 0
BATCH_SIZE = 5
BATCH_LENGTH = 400000
BUFFER_SIZE = 400
LSTM_UNITS = 64
NUM_CLASSES = 2
EPOCHS = 10
EMBEDDING_DIM = 60

NUM_DIMENSIONS = 300
MAX_SEQUENCE_LENGTH = 180
MAX_FILES_PER_USER  = 3200
MIN_SAMPLE_LENGTH = 5
#This is how many tweets the user has. The program will find the max number of Tweets that any user has
#and set this variable to that. Then, any users who don't have enough tweets, the values will be padded with zero
TENSOR_WIDTH = -1
MAX_TENSOR_LENGTH = MAX_SEQUENCE_LENGTH * MAX_FILES_PER_USER
NUM_LOADED_USERS = 0

MAKE_FILES = 0
VOCAB_DIFF = 0
COUNT_CHARS = 0
depression_path = ""
control_path = ""

if len(sys.argv) != 3:
    print('usage: python count_freqs.py "search term" "file type"')
    exit()

#standard vocabs
if sys.argv[2] == 's':
    depression_path = 'vocab/depression.vocab'
    control_path = 'vocab/control.vocab'
#other vocab 'stopwords removed'
elif sys.argv[2] == 'o':
    depression_path = 'vocab/depression_nostop.vocab'
    control_path = 'vocab/control_nostop.vocab'

if depression_path == "" or control_path == "":
    print("Error: no file paths defined!")
    exit()

if os.path.isfile(depression_path) and os.path.isfile(control_path) and MAKE_FILES == 0 and VOCAB_DIFF == 0 and COUNT_CHARS == 0:
    complete_match_d = 0
    complete_match_c = 0
    freqs_d = 0
    matches_d = []
    freqs_c = 0
    matches_c = []
    search = sys.argv[1]
    with open(depression_path, 'r') as f:
        for line in f:
            line = line.split(':')
            if " " not in line[0] and '\n' not in line[0]:
                line[1] = int(line[1].replace(',\n',''))
                #print(line)
                if search == line[0]:
                    complete_match_d += line[1]
                if search in line[0]:
                    freqs_d += line[1]
                    matches_d.append([line[0], line[1]])
    print("search term:'", search, "' appeared ", freqs_d, " times and a complete match was found: ",complete_match_d," in ", depression_path)
    with open(control_path, 'r') as f:
        for line in f:
            line = line.split(':')
            if " " not in line[0] and '\n' not in line[0]:
                line[1] = int(line[1].replace(',\n',''))
                #print(line)
                if search == line[0]:
                    complete_match_c += line[1]
                if search in line[0]:
                    freqs_c += line[1]
                    matches_c.append([line[0], line[1]])
    print("search term:'", search, "' was counted ", freqs_c, " times and a complete match was found: ",complete_match_c," in ", control_path)
    print('\n\n')
    print("Matches in Depression >5")
    for k,v in matches_d:
        if v < 5:
            break
        else:
            print("[",k,",",v,"]", end="")
    print("\n\n")
    print("Matches in Control >5")
    for k,v in matches_c:
        if v < 5:
            break
        else:
            print("[",k,",",v,"]", end="")
    #print("found terms in control: ", matches_c)
    print("\n")
    exit()

print("could not find the vocab files. creating vocab files.")
#s = word_tokenize("hello my dear friend should we go chase some dear friend")
dict_depression = defaultdict(int)
dict_control    = defaultdict(int)

def add_tokens_to_dict(tokens, in_dict):
    wordcount = in_dict
    for token in tokens:
        if token not in wordcount:
            wordcount[token] = 1
        else:
            wordcount[token] += 1
    return wordcount

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
depression_lengths = []
dep_len_before_pp = []
control_lengths = []
con_len_before_pp = []
FILE_STRINGS = []
FILE_STRINGS_CONTROL = []
data_samples_ommited = 0
count = 0
max_len = 0
print("Reading Depression Users")
for files in FILE_NAMES_DEPRESSION:
    string_user = []
    for file in files:
        tweet = ""
        all_lines = ""
        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                all_lines += line
            f.close()
        dep_len_before_pp.append(len(all_lines))
        pre_processed_tokens = ppc.pre_process_data(all_lines)
        for ppt in pre_processed_tokens:
            string_user.append(ppt)
    if len(string_user) > max_len:
        max_len = len(string_user)
    if len(string_user) < MIN_SAMPLE_LENGTH:
        data_samples_ommited = data_samples_ommited + 1
        continue
    if len(string_user) < MAX_TENSOR_LENGTH:
        depression_lengths.append(len(string_user))
        #padding not required for vocab counting
        #string_user += [''] * (MAX_TENSOR_LENGTH - len(string_user))
    count = count + 1
    FILE_STRINGS.append(string_user)
    if DEBUG_MODE == 1 and count > 10:
        break
NUM_LOADED_USERS += count
count = 0
print("Reading Control Users")
for files in FILE_NAMES_CONTROL:
    string_user = []
    for file in files:
        tweet = ""
        all_lines = ""
        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                all_lines += line
            f.close()
        con_len_before_pp.append(len(all_lines))
        pre_processed_tokens = ppc.pre_process_data(all_lines)
        for ppt in pre_processed_tokens:
            string_user.append(ppt) 
    if len(string_user) > max_len:
        max_len = len(string_user)
    if len(string_user) < MIN_SAMPLE_LENGTH:
        data_samples_ommited = data_samples_ommited + 1
        continue
    if len(string_user) < MAX_TENSOR_LENGTH:
        control_lengths.append(len(string_user))
        #padding not required for vocab counting
        #string_user += [''] * (MAX_TENSOR_LENGTH - len(string_user))
    count = count + 1
    FILE_STRINGS_CONTROL.append(string_user)
    if DEBUG_MODE == 1 and count > 10:
       break
NUM_LOADED_USERS += count
print("Depression Chars (before preprocess): ", sum(dep_len_before_pp))
print("after preprocess: ", sum(depression_lengths))
print("Control Chars (before preprocess): ", sum(con_len_before_pp))
print("after preprocess: ", sum(control_lengths))
print("Num Users: ", NUM_LOADED_USERS)
print("Max user length (chars): ", max_len)
max_token_length = 0

for user_tokens in FILE_STRINGS:
    add_tokens_to_dict(user_tokens, dict_depression)
    if len(user_tokens) > max_token_length:
        max_token_length = len(user_tokens)
for user_tokens in FILE_STRINGS_CONTROL:
    add_tokens_to_dict(user_tokens, dict_control)
    if len(user_tokens) > max_token_length:
        max_token_length = len(user_tokens)
print("Max token length: ", max_token_length)
if VOCAB_DIFF == 1:
    with open('vocab/freq_comparisons.vocab', 'w') as f:
        for k in dict_depression:
            print(k, " " , dict_control[k])
            if k in dict_control:
                out = str(k) + ": " + str(value-dict_control[k]) + ',' + '\n'
                f.write(out)
        f.close()
dict_depression = sorted(dict_depression.items(), key=lambda k_v: k_v[1], reverse=True)
dict_control    = sorted(dict_control.items(), key=lambda k_v: k_v[1], reverse=True)
if MAKE_FILES == 1:
    with open(depression_path, 'w') as f:
        for k, value in dict_depression:
            out = str(k) + ": " + str(value) + ',' + '\n'
            f.write(out)
        f.close()
    with open(control_path, 'w') as f:
        for k, value in dict_control:
            out = str(k) + ": " + str(value) + ',' + '\n'
            f.write(out)
        f.close()

print("done")
