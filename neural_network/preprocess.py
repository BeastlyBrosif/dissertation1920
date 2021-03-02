import nltk
from nltk.tokenize import word_tokenize
import sys
import re
import stop_words as s_w
import string
import datetime


def pre_process_data(line):
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
	#remove punctuation
	line = line.translate(str.maketrans('','',string.punctuation + "’"))
	#remove numbers from the string
	line = ''.join([w for w in line if not w.isdigit()])
	#tokenize the string
	line = word_tokenize(line)
	output_tokens = []
	for word in line:
		if word in s_w.countries:# and 1 == 0:
			word = word.replace(word, 'COUNTRY')
		if word in s_w.cities:# and 1 == 0:
			word = word.replace(word, 'LOCATION')
		letter = word[0]
		#convert first char to an int
		num = ord(letter)
		if num >= 97 and num <= 100:
			if word in s_w.names_AD:
			    word = word.replace(word, 'NAME')
		elif num >= 101 and num <= 103:
			if word in s_w.names_EG:
			    word = word.replace(word, 'NAME')
		elif num >= 104 and num <= 108:
			if word in s_w.names_HL:
			    word = word.replace(word, 'NAME')
		elif num >= 109 and num <= 115:
			if word in s_w.names_MS:
			    word = word.replace(word, 'NAME')
		elif num >= 116 and num <= 122:
			if word in s_w.names_TZ:
			    word = word.replace(word, 'NAME')
	line = output_tokens.append(word)
	return line

