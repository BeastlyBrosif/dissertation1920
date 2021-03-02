from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()


BUFFER_SIZE = 10000
BATCH_SIZE = 32

checkpoint_path = "training/cp.ckpt"
checkpoint_dir  = os.path.dirname(checkpoint_path)

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)

test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))


model = tf.keras.Sequential([
	tf.keras.layers.Embedding(encoder.vocab_size, 64),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(1, activation='softmax')
	])
model.compile(loss='binary_crossentropy',
			  optimizer=tf.keras.optimizers.Adam(1e-4),
			  metrics=['accuracy'])

loss, acc = model.evaluate(test_dataset, verbose=1)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss,acc = model.evaluate(test_dataset, verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


def pad_to_size(vec, size):
	zeros = [0] * (size - len(vec))
	vec.extend(zeros)
	return vec

def sample_predict(sentence, pad):
	encoded_sample_pred_text = encoder.encode(sentence)
	if pad:
		encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
	encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
	predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
	return (predictions)

plot_graphs(history, 'accuracy')

while 1==1:
	text = input("type in a move review: ")
	print("Reviewing %s..." % (text))
	predictions = sample_predict(text, pad=False)
	if predictions >= 0.5:
		print("Positive")
	else:
		print("Negative") 
	print(predictions)
