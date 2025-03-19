# %%
import pandas as pd
import numpy as np
import time
from memory_profiler import memory_usage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from gensim.models import KeyedVectors
import psutil



df = pd.read_csv(r"DATA.csv", encoding='ISO-8859-1')
df = df.drop("1467810369",axis=1)

# %%
df.columns = ['polarity', 'date', 'query', 'user', 'text']

df.head()

# %%
df = df.drop(["query","date","user"],axis=1)

# %%
df.head()

# %%
df.tail()

# %%
print("Shape of the DataFrame:", df.shape)
print("\nPolarity Value Counts:")
print(df["polarity"].value_counts())
print("\nUnique Values in 'polarity' column:")
print(df["polarity"].unique())
print("\nMissing (null) Values in the DataFrame:")
print(df.isnull().sum())


# %%
df["polarity"] = df["polarity"].replace(4, 1)

# %%
print("Shape of the DataFrame:", df.shape)
print("\nPolarity Value Counts:")
print(df["polarity"].value_counts())
print("\nUnique Values in 'polarity' column:")
print(df["polarity"].unique())
print("\nMissing (null) Values in the DataFrame:")
print(df.isnull().sum())


# %%
X = df['text']
y = df['polarity']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_df = pd.DataFrame({'text': X_train, 'polarity': y_train})
val_df = pd.DataFrame({'text': X_val, 'polarity': y_val})
test_df = pd.DataFrame({'text': X_test, 'polarity': y_test})

# %%
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

url_pattern = re.compile(r'http\S+|www\S+')
mention_pattern = re.compile(r'@\w+')
hashtag_pattern = re.compile(r'#\w+')
special_char_pattern = re.compile(r'[^a-zA-Z\s]')
extra_whitespace_pattern = re.compile(r'\s+')

def preprocess_tweet(tweet):
    tweet = url_pattern.sub('', tweet)
    tweet = mention_pattern.sub('', tweet)
    tweet = hashtag_pattern.sub('', tweet)
    tweet = special_char_pattern.sub('', tweet.lower())

    tokens = [lemmatizer.lemmatize(word) for word in tweet.split() if word not in stop_words]

    return extra_whitespace_pattern.sub(' ', ' '.join(tokens)).strip()


# %% [markdown]
# Remove URLs
# 
# Remove Mentions
# 
# Remove Hashtags
# 
# Remove Special Characters and Lowercasing     
# 
# Remove Stopwords , Tokenization , Lemmatization
# 
# Remove Extra Whitespaces

# %%
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# %%
sample_size = 0.5

train_df = train_df.groupby(train_df.iloc[:, 1]).apply(lambda x: x.sample(frac=sample_size, random_state=42)).reset_index(drop=True)
test_df = test_df.groupby(test_df.iloc[:, 1]).apply(lambda x: x.sample(frac=sample_size, random_state=42)).reset_index(drop=True)
val_df = val_df.groupby(val_df.iloc[:, 1]).apply(lambda x: x.sample(frac=sample_size, random_state=42)).reset_index(drop=True)

print(train_df.iloc[:, 1].value_counts(normalize=True))
print(test_df.iloc[:, 1].value_counts(normalize=True))
print(val_df.iloc[:, 1].value_counts(normalize=True))

# %%
train_df['text'] = train_df['text'].apply(preprocess_tweet)
val_df['text'] = val_df['text'].apply(preprocess_tweet)
test_df['text'] = test_df['text'].apply(preprocess_tweet)

# %%
train_texts, train_labels = train_df["text"], train_df["polarity"]
test_texts, test_labels = test_df["text"], test_df["polarity"]
val_texts, val_labels = val_df["text"], val_df["polarity"]

# %%
train_texts = train_texts.astype(str)
test_texts = test_texts.astype(str)
val_texts = val_texts.astype(str)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)

# %%
total_words = sum(tokenizer.word_counts.values())
sorted_counts = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)

cumulative_freq = 0
threshold = 0.95 * total_words
num_words = 0

for word, count in sorted_counts:
    cumulative_freq += count
    num_words += 1
    if cumulative_freq >= threshold:
        break

print(f"num_words to cover 95% of the word occurrences: {num_words}")

# %% [markdown]
# The cell above help us identify number of words in the below cell

# %%
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_texts)

vocab_size = len(tokenizer.word_index) + 1
print(f"Final vocabulary size: {vocab_size}")

# %%
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)

# %% [markdown]
# Convert Text to sequence of Numeric Format

# %%
sequence_lengths = [len(seq) for seq in train_sequences]

print(f"Max length: {np.max(sequence_lengths)}")

# %% [markdown]
# The cell above help us identify sequence_lengths in the below cell

# %%
max_sequence_length = np.max(sequence_lengths)
train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)
val_padded = pad_sequences(val_sequences, maxlen=max_sequence_length)

# %% [markdown]
# Make all input data of the same shape by padding or truncating

# %%
trainable_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=50, input_length=max_sequence_length, trainable=True),
    LSTM(128, dropout=0.4, recurrent_dropout=0),
    Dense(1, activation="sigmoid")
])

trainable_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
trainable_model.summary()

# %%
checkpoint_dir = "model_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'best_weights.h5'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# %%
trainable_model.fit(train_padded, train_labels, 
                    epochs=50, batch_size=64, 
                    validation_data=(val_padded, val_labels),
                    callbacks=[early_stop, model_checkpoint])

# %%
# LSTM Layer Parameters
lstm_weights = trainable_model.layers[1].get_weights()
W_i, U_i, b_i = lstm_weights[0][:, :128], lstm_weights[1][:, :128], lstm_weights[2][:128]  # Input gate
W_f, U_f, b_f = lstm_weights[0][:, 128:256], lstm_weights[1][:, 128:256], lstm_weights[2][128:256]  # Forget gate
W_c, U_c, b_c = lstm_weights[0][:, 256:384], lstm_weights[1][:, 256:384], lstm_weights[2][256:384]  # Cell state
W_o, U_o, b_o = lstm_weights[0][:, 384:], lstm_weights[1][:, 384:], lstm_weights[2][384:]  # Output gate

# Dense Layer Parameters
dense_weights, dense_bias = trainable_model.layers[2].get_weights()

# %% [markdown]
# Extract Model Weights for the forward pass

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def LSTM_forward(inputs, mask=None):
    timesteps, input_dim = inputs.shape
    h_t = np.zeros((128,))  # Initialize hidden state
    c_t = np.zeros((128,))  # Initialize cell state

    for t in range(timesteps):
        if mask is not None and mask[t] == 0:
            continue  # Skip this timestep if masked

        x_t = inputs[t]

        # Input gate
        i_t = sigmoid(np.dot(x_t, W_i) + np.dot(h_t, U_i) + b_i)
        # Forget gate
        f_t = sigmoid(np.dot(x_t, W_f) + np.dot(h_t, U_f) + b_f)
        # Cell state update
        c_hat_t = np.tanh(np.dot(x_t, W_c) + np.dot(h_t, U_c) + b_c)
        c_t = f_t * c_t + i_t * c_hat_t
        # Output gate
        o_t = sigmoid(np.dot(x_t, W_o) + np.dot(h_t, U_o) + b_o)
        # Hidden state update
        h_t = o_t * np.tanh(c_t)

    # Dense layer forward pass
    dense_output = sigmoid(np.dot(h_t, dense_weights) + dense_bias)
    return dense_output

def predict(batch):
    # Pass the batch through the embedding layer
    embeddings = trainable_model.layers[0](batch).numpy()
    mask = (batch != 0).astype(int)  # Create mask (assuming 0 is the padding token)

    predictions = []
    for i in range(embeddings.shape[0]):  # Process each sequence
        pred = LSTM_forward(embeddings[i], mask=mask[i])
        predictions.append(pred)

    return np.array(predictions)


# %%
def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)

sample_data = test_padded

threshold = 0.5

# Model predictions
start_time = time.time()
start_memory = get_memory_usage()
model_preds = trainable_model.predict(sample_data)
model_classes = (model_preds >= threshold).astype(int)
model_memory = get_memory_usage() - start_memory
model_time = time.time() - start_time

# Custom predictions
start_time = time.time()
start_memory = get_memory_usage()
custom_preds = predict(sample_data)
custom_classes = (custom_preds >= threshold).astype(int)
custom_memory = get_memory_usage() - start_memory
custom_time = time.time() - start_time

print("Model Classes:", model_classes.flatten())
print("Custom Classes:", custom_classes.flatten())

accuracy = np.mean(model_classes == custom_classes)
print(f"Accuracy of Custom Predictions: {accuracy * 100:.2f}%")

print(f"Custom Function Time: {custom_time:.4f} seconds")
print(f"Model Function Time: {model_time:.4f} seconds")
print(f"Custom Function Memory: {custom_memory:.4f} MB")
print(f"Model Function Memory: {model_memory:.4f} MB")

# %% [markdown]
# The custom implementation achieves high accuracy, nearly identical to the trained model.The minor differences (accuracy less than 100%) may arise due to:
# 
# Numerical precision differences between NumPy operations and TensorFlow optimizations.
# 
# The manual sigmoid and tanh functions might have slight deviations compared to their TensorFlow counterparts.

# %% [markdown]
# 1) TensorFlow Optimization:
# 
# TensorFlow is highly optimized for GPU and parallelized computation, enabling rapid forward passes for batches of data. It leverages cuDNN, to process LSTM computations efficiently.
# 
# 2) Custom Function:
# 
# The custom function processes the input sequentially for each timestep and sequence, leading to significant overhead.
# It runs on the CPU using NumPy, which is not optimized for deep learning computations.
# Masking is implemented manually, adding additional computation for each timestep.

# %% [markdown]
# 1) TensorFlow's Overheads:
# 
# TensorFlow allocates additional memory for internal graph representations, tensors, and other intermediate computations to optimize performance and enable gradient computations.
# 
# 2) Custom Function :
# The custom function only computes the forward pass without the overhead of maintaining computational graphs or intermediate states.
# It uses NumPy arrays, which have lower memory overhead compared to TensorFlow's tensor objects.

# %%
trainable_acc = trainable_model.evaluate(test_padded, test_labels, batch_size=64)
print(f"Trainable Embeddings Model Accuracy: {trainable_acc[1]:.4f}")

# %%
y_pred_probs = trainable_model.predict(test_padded, batch_size=64)
y_pred = (y_pred_probs > 0.5).astype("int32")

# %%
conf_mat = confusion_matrix(test_labels, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["Class 0", "Class 1"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Trainable Embeddings Model")
plt.show()

# %%
report = classification_report(test_labels, y_pred, target_names=["Class 0", "Class 1"])
print("Classification Report:\n")
print(report)

# %%
model_path = r"C:\Users\Essam\Desktop\DL Assignment2\Word2vec\word2vec-google-news-300\word2vec-google-news-300.gz"
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# %%
unmatched_count = 0
matched_count = 0

for word in tokenizer.word_index.keys():
    if word in word2vec_model:
        matched_count += 1
    else:
        unmatched_count += 1

print(f"Matched tokens: {matched_count}")
print(f"Unmatched tokens (OOV): {unmatched_count}")


# %% [markdown]
# number of words not covered by word2vec embedding

# %%
def build_embedding_matrix(word_index, embedding_dim=300):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model:
            embedding_matrix[i] = word2vec_model[word]
    return embedding_matrix

embedding_matrix = build_embedding_matrix(tokenizer.word_index)

pretrained_model = Sequential([
    Embedding(input_dim=len(embedding_matrix),
              output_dim=300, input_length=max_sequence_length,
              weights=[embedding_matrix], trainable=False),
    LSTM(128, dropout=0.2, recurrent_dropout=0),
    Dense(1, activation="sigmoid")
])

pretrained_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
pretrained_model.summary()

# %%
checkpoint_dir = "model_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

early_stop_pretrained  = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)
model_checkpoint_pretrained  = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'best_weights_pretrained.weights.h5'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# %%
pretrained_model.fit(train_padded, train_labels,
                    epochs=50, batch_size=64,
                    validation_data=(val_padded, val_labels),
                    callbacks=[early_stop_pretrained, model_checkpoint_pretrained])

# %%
# Extract LSTM Layer Parameters from the pretrained model
lstm_weights = pretrained_model.layers[1].get_weights()
W_i, U_i, b_i = lstm_weights[0][:, :128], lstm_weights[1][:, :128], lstm_weights[2][:128]  # Input gate
W_f, U_f, b_f = lstm_weights[0][:, 128:256], lstm_weights[1][:, 128:256], lstm_weights[2][128:256]  # Forget gate
W_c, U_c, b_c = lstm_weights[0][:, 256:384], lstm_weights[1][:, 256:384], lstm_weights[2][256:384]  # Cell state
W_o, U_o, b_o = lstm_weights[0][:, 384:], lstm_weights[1][:, 384:], lstm_weights[2][384:]  # Output gate

# Extract Dense Layer Parameters from the pretrained model
dense_weights, dense_bias = pretrained_model.layers[2].get_weights()

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def LSTM_forward(inputs, mask=None):
    timesteps, input_dim = inputs.shape
    h_t = np.zeros((128,))  # Initialize hidden state
    c_t = np.zeros((128,))  # Initialize cell state

    for t in range(timesteps):
        if mask is not None and mask[t] == 0:
            continue  # Skip this timestep if masked

        x_t = inputs[t]

        # Input gate
        i_t = sigmoid(np.dot(x_t, W_i) + np.dot(h_t, U_i) + b_i)
        # Forget gate
        f_t = sigmoid(np.dot(x_t, W_f) + np.dot(h_t, U_f) + b_f)
        # Cell state update
        c_hat_t = np.tanh(np.dot(x_t, W_c) + np.dot(h_t, U_c) + b_c)
        c_t = f_t * c_t + i_t * c_hat_t
        # Output gate
        o_t = sigmoid(np.dot(x_t, W_o) + np.dot(h_t, U_o) + b_o)
        # Hidden state update
        h_t = o_t * np.tanh(c_t)

    # Dense layer forward pass
    dense_output = sigmoid(np.dot(h_t, dense_weights) + dense_bias)
    return dense_output
def predict(batch):
    # Pass the batch through the embedding layer of the pretrained model
    embeddings = pretrained_model.layers[0](batch).numpy()
    mask = (batch != 0).astype(int)  # Create mask (assuming 0 is the padding token)

    predictions = []
    for i in range(embeddings.shape[0]):  # Process each sequence
        pred = LSTM_forward(embeddings[i], mask=mask[i])
        predictions.append(pred)

    return np.array(predictions)


# %%
def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)

threshold = 0.5

sample_data = test_padded[:1000]

# Model predictions
start_time = time.time()
start_memory = get_memory_usage()
model_preds = pretrained_model.predict(sample_data)
model_classes = (model_preds >= threshold).astype(int)
model_memory = get_memory_usage() - start_memory
model_time = time.time() - start_time

# Custom predictions
start_time = time.time()
start_memory = get_memory_usage()
custom_preds = predict(sample_data)
custom_classes = (custom_preds >= threshold).astype(int)
custom_memory = get_memory_usage() - start_memory
custom_time = time.time() - start_time

print("Model Classes:", model_classes.flatten())
print("Custom Classes:", custom_classes.flatten())

accuracy = np.mean(model_classes == custom_classes)
print(f"Accuracy of Custom Predictions: {accuracy * 100:.2f}%")

print(f"Custom Function Time: {custom_time:.4f} seconds")
print(f"Model Function Time: {model_time:.4f} seconds")
print(f"Custom Function Memory: {custom_memory:.4f} MB")
print(f"Model Function Memory: {model_memory:.4f} MB")


# %%
test_loss, test_acc = pretrained_model.evaluate(test_padded, test_labels)
print(f"Test Accuracy: {test_acc}")

# %%
test_preds = pretrained_model.predict(test_padded)
test_preds = (test_preds > 0.5).astype(int)

# %%
conf_mat = confusion_matrix(test_labels, test_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["Class 0", "Class 1"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Trainable Embeddings Model")
plt.show()

# %%
report = classification_report(test_labels, test_preds, target_names=["Negative", "Positive"])
print(report)

# %% [markdown]
# ### Both Models Trained and Pre-trained achieving simillar accuracies but the pretrained model is much faster as it is using pretrained embeddings 


