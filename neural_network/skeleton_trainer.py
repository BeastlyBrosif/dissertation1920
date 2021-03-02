
import tensorflow as tf

BATCH_SIZE = 24
LSTM_UNITS = 64
NUM_CLASSES = 2
BUFFER_SIZE = 10000

NUM_DIMENSIONS = 300
MAX_SEQUENCE_LENGTH = 180

#This is how many tweets the user has. The program will find the max number of Tweets that any user has
#and set this variable to that. Then, any users who don't have enough tweets, the values will be padded with zero
TENSOR_WIDTH = -1

#load the data into a list
#preprocess the data
#transform the data into a tensor

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

TRAIN_DATA_DIR = ""
TEST_DATA_DIR  = ""

def labeler(exapmle, index):
	return example, tf.cast(index, tf.int64)

#list of all files being loaded
FILE_NAMES = []
labeled_data_sets = []
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


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'Hello TensorFlow.'
encoded_string = encoder.encode(sample_string)
original_string = encoder.decode(encoded_string)

BUFFER_SIZE = 10000
BATCH_SIZE = 32

checkpoint_path = "training/cp.ckpt"
checkpoint_dir  = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)







train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)

test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
input_data = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SEQUENCE_LENGTH])

data = tf.Variable(tf.zeros([BATCH_SIZE, MAX_SEQUENCE_LENGTH, NUM_DIMENSIONS]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtupe=tf.float32)

weight = tf.Variable(tf.truncated_normal([LSTM_UNITS, NUM_CLASSES]))
bias   = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
value  = tf.transpose(value, [1, 0, 2])
last   = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = tf.matmul(last, weight) + bias

correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
optimizer = tf.train.AdamOptimizer().minimise(loss)

tf.nn.rnn_cell.BasicLSTMCell

