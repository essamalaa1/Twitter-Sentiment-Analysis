# Twitter Sentiment Analysis with LSTM
![Alt text](https://github.com/essamalaa1/Twitter-Sentiment-Analysis/blob/main/images.png)

This project performs sentiment analysis on Twitter data using Long Short-Term Memory (LSTM) networks. It explores two approaches:
1.  Training an LSTM model with its own trainable embedding layer.
2.  Training an LSTM model using pre-trained Word2Vec (Google News 300) embeddings.

Additionally, for both models, a custom Python implementation of the LSTM forward pass is demonstrated and compared against the Keras model's predictions for accuracy, execution time, and memory usage.

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
- [Models](#models)
  - [Model 1: Trainable Embeddings LSTM](#model-1-trainable-embeddings-lstm)
  - [Model 2: Pre-trained Word2Vec LSTM](#model-2-pre-trained-word2vec-lstm)
- [Custom LSTM Forward Pass](#custom-lstm-forward-pass)
- [Results](#results)
- [Requirements](#requirements)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)

## Dataset
The project uses a Twitter dataset named `DATA.csv`.
- **Source**: The notebook reads `DATA.csv` with `ISO-8859-1` encoding. It appears to be a standard sentiment analysis dataset (likely Sentiment140 or similar).
- **Columns Used**: After initial cleaning, the primary columns used are `polarity` and `text`.
- **Labels**:
    - `0`: Negative sentiment
    - `4`: Positive sentiment (mapped to `1` during preprocessing)
- The dataset is split into training (80%), validation (10%), and test (10%) sets. A 50% sample of each class is then taken for these splits to manage computational resources.

## Features
- **Text Preprocessing**:
    - Removal of URLs, mentions (@username), and hashtags (#topic).
    - Conversion to lowercase.
    - Removal of special characters.
    - Tokenization.
    - Stopword removal (NLTK English stopwords).
    - Lemmatization (NLTK WordNetLemmatizer).
    - Removal of extra whitespaces.
- **Deep Learning Models**:
    - Two LSTM-based models for binary sentiment classification.
- **Embedding Layers**:
    - Custom trainable embedding layer.
    - Pre-trained Google News Word2Vec (300 dimensions) embedding layer.
- **Custom LSTM Forward Pass**:
    - Manual implementation of the LSTM and Dense layer forward pass using NumPy.
    - Comparison of custom implementation with Keras model predictions in terms of:
        - Accuracy
        - Execution Time
        - Memory Usage
- **Model Evaluation**:
    - Accuracy.
    - Confusion Matrix.
    - Classification Report (precision, recall, F1-score).
- **Training Enhancements**:
    - `EarlyStopping` to prevent overfitting.
    - `ModelCheckpoint` to save the best model weights.

## Methodology
1.  **Data Loading & Initial Cleaning**: Load `DATA.csv`, drop irrelevant columns, and rename essential columns.
2.  **Data Preprocessing**:
    - Map sentiment label `4` to `1`.
    - Apply the `preprocess_tweet` function to clean the text data.
3.  **Data Splitting**: Split data into training, validation, and test sets. Further sample these sets.
4.  **Tokenization & Padding**:
    - Use `keras.preprocessing.text.Tokenizer` to convert text to sequences of integers. Vocabulary size is determined to cover 95% of word occurrences.
    - Pad sequences to a maximum length using `keras.preprocessing.sequence.pad_sequences`.
5.  **Model 1 (Trainable Embeddings)**:
    - Define an LSTM model with a trainable `Embedding` layer.
    - Compile and train the model with `EarlyStopping` and `ModelCheckpoint`.
    - Evaluate the model on the test set.
    - Extract weights and implement/test the custom LSTM forward pass.
6.  **Model 2 (Pre-trained Word2Vec Embeddings)**:
    - Load the pre-trained Word2Vec (Google News 300) model.
    - Create an embedding matrix mapping words in the vocabulary to their Word2Vec vectors.
    - Define an LSTM model with a non-trainable `Embedding` layer initialized with the Word2Vec matrix.
    - Compile and train the model.
    - Evaluate the model on the test set.
    - Extract weights and implement/test the custom LSTM forward pass.
7.  **Comparison**: Analyze the performance and characteristics of both models.

## Models

### Model 1: Trainable Embeddings LSTM
- **Embedding Layer**: `Embedding(input_dim=vocab_size, output_dim=50, trainable=True)`
- **LSTM Layer**: `LSTM(128, dropout=0.4)`
- **Output Layer**: `Dense(1, activation="sigmoid")`
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

### Model 2: Pre-trained Word2Vec LSTM
- **Embedding Layer**: `Embedding(input_dim=len(embedding_matrix), output_dim=300, weights=[embedding_matrix], trainable=False)` (using Google News 300d Word2Vec)
- **LSTM Layer**: `LSTM(128, dropout=0.2)`
- **Output Layer**: `Dense(1, activation="sigmoid")`
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

## Custom LSTM Forward Pass
For both models, after training with Keras, the weights of the LSTM and Dense layers are extracted. A custom Python function (`LSTM_forward`) is implemented using NumPy to replicate the forward pass mechanism of a single LSTM cell followed by the dense layer. This includes:
- Input, Forget, Cell, and Output gate calculations.
- Cell state and hidden state updates.
- Sigmoid and tanh activation functions (NumPy implementations).
- A `predict` wrapper function handles batch processing and the embedding layer lookup (using the Keras embedding layer for convenience).

The predictions from this custom implementation are compared to the Keras `model.predict()` output. This serves as an educational exercise to understand the internals of an LSTM.

**Key Findings from Custom Forward Pass:**
- **Accuracy**: The custom implementation achieves very high accuracy (close to 100%) when compared to the Keras model's predictions, indicating a correct understanding and implementation of the forward pass.
- **Time**: The Keras `predict` function is significantly faster due to optimized C++ backends and parallelization, while the custom NumPy implementation is slower as it processes sequences and timesteps iteratively in Python.
- **Memory**: The custom function often shows lower memory usage for the computation itself, as it doesn't carry the overhead of TensorFlow's graph management and other utilities during a simple forward pass. However, TensorFlow's memory usage during `predict` can also be efficient for large batches due to its optimizations.

## Results
- **Trainable Embeddings Model Accuracy**: ~79.09% on the test set.
- **Pre-trained Word2Vec Model Accuracy**: ~79.27% on the test set.
- Both models achieve similar overall accuracy on this task.
- The model with pre-trained Word2Vec embeddings trains faster per epoch because its embedding layer is not trainable, reducing the number of parameters to update during backpropagation. (e.g., ~65s/epoch for Word2Vec model vs. ~155s/epoch for trainable embeddings model on the first epoch).
- The custom LSTM forward pass implementation validated the mathematical operations within the LSTM cell.

## Requirements
- Python 3.8+
- pandas
- numpy
- tensorflow (tested with a version compatible with Keras Sequential API, e.g., 2.x)
- scikit-learn
- matplotlib
- nltk
- gensim
- psutil

A `requirements.txt` file can be generated using:
`pip freeze > requirements.txt`
(It's recommended to do this in a clean virtual environment with only the project dependencies installed.)

## Setup & Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or install manually:
    # pip install pandas numpy tensorflow scikit-learn matplotlib nltk gensim psutil
    ```
4.  **Download NLTK resources:**
    Open a Python interpreter and run:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4') # WordNetLemmatizer might require this
    ```
5.  **Download Pre-trained Word2Vec Model:**
    - Download the Google News 300-dimensional Word2Vec model (`word2vec-google-news-300.gz`). A common source is [Kaggle](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300) or other public repositories.
    - Place the `word2vec-google-news-300.gz` file in a directory accessible by the notebook. The notebook currently has a hardcoded path: `r"C:\Users\Essam\Desktop\DL Assignment2\Word2vec\word2vec-google-news-300\word2vec-google-news-300.gz"`. You will need to **update this path** in cell 37 to where you've saved the model.

6.  **Dataset:**
    - Ensure the `DATA.csv` file is in the same directory as the Jupyter notebook, or update the path in the notebook accordingly.

## Usage
1.  Ensure all setup steps are completed, especially the NLTK resource downloads and Word2Vec model path update.
2.  Open and run the `Twitter_sentiment_analysis.ipynb` notebook using Jupyter Lab or Jupyter Notebook.
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
3.  Execute the cells sequentially. The notebook will:
    - Load and preprocess the data.
    - Train and evaluate the LSTM model with trainable embeddings.
    - Demonstrate and compare the custom LSTM forward pass for this model.
    - Train and evaluate the LSTM model with pre-trained Word2Vec embeddings.
    - Demonstrate and compare the custom LSTM forward pass for this second model.
    - Display confusion matrices and classification reports.
  
**Note:** You need to create the `path_to_word2vec/` directory (or choose your own path) and place the downloaded Word2Vec model there. Update the path in the notebook. The `model_checkpoints/` directory will be created automatically by the script.
