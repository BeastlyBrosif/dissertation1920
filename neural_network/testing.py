from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import tensorflow_datasets as tfds


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print ('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print ('The original string: "{}"'.format(original_string))

BUFFER_SIZE = 10000
BATCH_SIZE = 32

checkpoint_path = "training/cp.ckpt"
checkpoint_dir  = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)

test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

print("HERE!")



model = tf.keras.Sequential([
	tf.keras.layers.Embedding(encoder.vocab_size, 64),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

model.compile(loss='binary_crossentropy',
			  optimizer=tf.keras.optimizers.Adam(1e-4),
			  metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, 
					validation_data=test_dataset,
					validation_steps=30,
					callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

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

model.summary()



text = "The movie was bad, the animation was bad, everything was terrible!"
predictions = sample_predict(text, pad=False)
print(predictions)