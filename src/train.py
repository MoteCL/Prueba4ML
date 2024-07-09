import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load preprocessed data
df = pd.read_csv('data/preprocessed/preprocessed_data.csv')

# Ensure all data are strings
df['transformed_text'] = df['transformed_text'].astype(str)

# Prepare data for training
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(df['transformed_text'].values)
X = tokenizer.texts_to_sequences(df['transformed_text'].values)
X = pad_sequences(X)
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model 

model = Sequential()
model.add(Embedding(5000, 128))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# Save the model
model.save('models/model.h5')
