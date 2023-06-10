import requests
import json
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# API endpoint for card data
api_url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

# Parameters for API request
params = {
    "fname": "name",
    "num": 1000  # Number of cards to retrieve
}

# Send API request
response = requests.get(api_url, params=params)
response_data = json.loads(response.text)

# Check if 'data' key exists in the response JSON
if "data" in response_data:
    data = response_data["data"]
    if len(data) > 0:
        # Extract card names from API response
        card_names = [card["name"] for card in data]
    else:
        print("Error: No card data found in the API response.")
        card_names = []
else:
    print("Error: 'data' key not found in the API response.")
    card_names = []

# Preprocess the data
all_chars = list(set(' '.join(card_names)))  # All unique characters in the card names
num_chars = len(all_chars)
char_to_index = {char: index for index, char in enumerate(all_chars)}
index_to_char = {index: char for index, char in enumerate(all_chars)}

# Generate training data
max_sequence_length = 20  # Maximum length of input sequence
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
model.add(LSTM(128, input_shape=(max_sequence_length, 1)))
model.add(Dense(num_chars, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(np.expand_dims(X, axis=-1), y, batch_size=128, epochs=50)

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
