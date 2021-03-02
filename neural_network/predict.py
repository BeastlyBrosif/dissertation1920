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

