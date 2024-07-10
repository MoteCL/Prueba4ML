import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

# Load and preprocess the data
data_path = os.path.join(os.path.dirname(__file__), '../data/preprocessed/preprocessed_data.csv')
data = pd.read_csv(data_path)

texts = data['transformed_text'].values.astype(str)
labels = data['target'].values

# Tokenizer setup
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = 100
X = pad_sequences(sequences, maxlen=max_sequence_length)
y = labels

# Build the model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_sequence_length))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# Save the model
model_path = os.path.join(os.path.dirname(__file__), '../models/model.h5')
model.save(model_path)

# Save the tokenizer
tokenizer_path = os.path.join(os.path.dirname(__file__), '../models/tokenizer.pkl')
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
