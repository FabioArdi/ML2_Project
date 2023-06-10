import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import CuDNNLSTM, Dense, Dropout, BatchNormalization

# API endpoint
url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

# Request parameters
params = {

}

word_list = dict()
subtype_list = dict()
char_list = dict()
card_number_listed = 0

try:
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200:
        cards = data["data"]

        card_names = [card["name"] for card in cards]
        card_descriptions = [card["desc"] for card in cards]
        card_types = [card["type"] for card in cards]

except requests.exceptions.RequestException as e:
    print("An error occurred:", e)

# for i in range(len(card_names)):
#    print(card_names[i])
#    print(card_types[i])
#    print()

print("Number of cards:", len(card_names))
print(card_names[:10])

vocab_size = len(card_names)
embedding_dim = 256
rnn_units = 1024
batch_size = 64

# Building the model structure
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
    tf.keras.layers.CuDNNLSTM(rnn_units,
                            return_sequences=True,
                            recurrent_initializer='glorot_uniform',
                            stateful=True),
    tf.keras.layers.Dense(vocab_size)
])

# Preprocessing the data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(card_names)
sequences = tokenizer.texts_to_sequences(card_names)
vocab_size = len(tokenizer.word_index) + 1

input_sequences = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Creating training data
x = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Splitting into training and validation sets
train_size = int(0.8 * len(x))
x_train, y_train = x[:train_size], y[:train_size]
x_val, y_val = x[train_size:], y[train_size:]

# Building the model structure
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_sequence_len-1),
    tf.keras.layers.CuDNNLSTM(rnn_units, return_sequences=True),
    tf.keras.layers.CuDNNLSTM(rnn_units),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=batch_size)

# Generating new card names
seed_text = "start_token"  # Provide a seed sequence or start token
num_generate = 10  # Number of card names to generate

for _ in range(num_generate):
    tokenized_seed = tokenizer.texts_to_sequences([seed_text])[0]
    padded_seed = tf.keras.preprocessing.sequence.pad_sequences([tokenized_seed], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(padded_seed, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
    print(seed_text)

