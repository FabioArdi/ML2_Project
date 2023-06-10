import requests
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout

# API endpoint
url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

# Request parameters
params = {

}

try:
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200:
        cards = data["data"]

        card_names = [card["name"] for card in cards]
        card_descriptions = [card["desc"] for card in cards]

except requests.exceptions.RequestException as e:
    print("An error occurred:", e)


# Preprocess the data
all_chars = list(set(' '.join(card_names)))  # All unique characters in the card names
num_chars = len(all_chars)
char_to_index = {char: index for index, char in enumerate(all_chars)}
index_to_char = {index: char for index, char in enumerate(all_chars)}

# Generate training data
max_sequence_length = 10  # Maximum length of input sequence
sequences = []
next_chars = []
for card_name in card_names:
    for i in range(len(card_name) - max_sequence_length):
        sequence = card_name[i:i + max_sequence_length]
        target_char = card_name[i + max_sequence_length]
        sequences.append([char_to_index[char] for char in sequence])
        next_chars.append(char_to_index[target_char])

# Convert training data to numpy arrays
X = np.array(sequences)
y = np.array(next_chars)

# Normalize input data
X = X / float(num_chars)

# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(max_sequence_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dense(num_chars, activation='softmax'))


# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(np.expand_dims(X, axis=-1), y, batch_size=128, epochs=2)

# Generate new card names
seed_sequence = random.choice(sequences)
generated_card_name = ''.join([index_to_char[index] for index in seed_sequence])

for _ in range(10):
    x_pred = np.expand_dims(seed_sequence, axis=0) / float(num_chars)
    prediction = model.predict(x_pred)[0]
    predicted_index = np.argmax(prediction)
    generated_char = index_to_char[predicted_index]
    generated_card_name += generated_char
    seed_sequence = seed_sequence[1:] + [predicted_index]

print("Generated Card Names:")
print(generated_card_name)

