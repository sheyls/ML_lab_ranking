import datetime

import numpy as np
import os
import re
from nltk.test.unit.lm.test_models import training_data
from sklearn.model_selection import train_test_split

import config
from gensim.models import KeyedVectors
from tqdm import tqdm
from tokenization import *
import tensorflow as tf
import tensorflow_models as tfm

def build_loinc_embedding(loinc_vocab):
    model_dir = "./model"

    matrix_save_path = os.path.join(model_dir, "loinc_embedding_matrix.npy")
    sz = loinc_vocab.shape[0]
    vocab_size = sz + 1  # +1 to account for padding token (index 0)
    embedding_dim = 200

    # Try to load existing artifacts
    if os.path.exists(matrix_save_path):
        print("Loading pre-built loinc embedding layer")
        embedding_matrix = np.load(matrix_save_path)
        print("Loaded.")
    else:
        max_sequence_length = 100  # for example

        model_path = os.path.join(config.DATA_BIOSENTVEC, 'BioWordVec_PubMed_MIMICIII_d200.vec.bin')
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print('Model successfully loaded')

        embedding_dim = model.vector_size

        print(f"With size {model.vector_size}")

        # Create the embedding matrix
        print("Creating loinc embedding...")
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for i, row in tqdm(
                loinc_vocab.iterrows(),
                total=len(loinc_vocab),
                desc="Building embeddings",
                colour="green",  # Change progress bar color
                dynamic_ncols=True  # Adjust width to terminal
        ):
            component_text = row['COMPONENT']

            token_embeddings = []
            for token in preprocess_sentence(component_text):
                if token in model:
                    token_embeddings.append(model[token])

            if token_embeddings:
                avg_embedding = np.mean(token_embeddings, axis=0)
            else:
                # If no tokens are found, use random initialization
                avg_embedding = np.random.normal(scale=0.6, size=(embedding_dim,))
            embedding_matrix[i] = avg_embedding

        del model
        os.makedirs(model_dir, exist_ok=True)
        np.save(matrix_save_path, embedding_matrix)

    # Now, create a Keras Embedding layer with these weights.
    # Set trainable=True if you wish to fine-tune the embeddings during training.
    (vocab_size, embedding_dim) = embedding_matrix.shape
    loinc_embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=1,
        trainable=True
    )

    return loinc_embedding_layer


def build_query_embedding(max_sequence_length):
    # Define save paths
    model_dir = "./model"
    embedding_save_path = os.path.join(model_dir, "query_embedding.keras")
    matrix_save_path = os.path.join(model_dir, "query_embedding_matrix.npy")
    tmp_save_path = os.path.join(model_dir, "tmp_query_embedding_matrix.npy")
    # Try to load existing artifacts
    if os.path.exists(matrix_save_path):
        print("Loading pre-built query embedding layer")
        embedding_matrix = np.load(matrix_save_path)
        print("Loaded.")
    else:
        model_path = os.path.join(config.DATA_BIOSENTVEC, 'BioWordVec_PubMed_MIMICIII_d200.vec.bin')
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        embedding_dim = model.vector_size
        print(model.vector_size) # 16545452 x 200?
        print('Model successfully loaded')

        print(str(model.index_to_key)[:100])
        # Initialize embedding matrix with zeros or random values
        vocab_size = len(model.index_to_key)  # Use model.wv.index_to_key for gensim <4.0.0
        embedding_matrix = np.memmap(tmp_save_path, dtype='float16', mode='w+',
                       shape=(vocab_size + 1, embedding_dim))
        print("Creating query embedding...")
        # Populate the matrix using the model's vocabulary
        for i, word in tqdm(enumerate(model.index_to_key, start=1), desc="Building embeddings", colour="green",
                            total=vocab_size):  # Start at index 1 (index 0 is padding)
            embedding_matrix[i] = model[word].astype(np.float16)

        del model
        np.save(matrix_save_path, embedding_matrix)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    # Now, create a Keras Embedding layer with these weights.
    # Set trainable=True if you wish to fine-tune the embeddings during training.
    (vocab_size, embedding_dim) = embedding_matrix.shape
    query_embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size + 1,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_sequence_length,
        trainable=True
    )
    return query_embedding_layer

class FakeEmbedding:
    def __init__(self):
        pass
    def __getitem__(self, item):
        return np.random.normal(size=(200,))
    def __contains__(self, item):
        return True

class Embedding:
    def __init__(self, loinc, debug=True, max_sentence_length=100):
        if not debug:
            model_path = os.path.join(config.DATA_BIOSENTVEC, 'BioWordVec_PubMed_MIMICIII_d200.vec.bin')
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        else:
            self.model = FakeEmbedding()
        self.loinc = loinc
        self.max_sentence_length = max_sentence_length
        self.pad_token = np.zeros((200,))

    def sentence2vecs(self, sentence):
        token_embeddings = []
        for token in preprocess_sentence(sentence):
            if token in self.model:
                token_embeddings.append(self.model[token])
        # Add padding if necessary
        current_length = len(token_embeddings)

        token_embeddings = token_embeddings[:self.max_sentence_length]

        if current_length < self.max_sentence_length:
            # Calculate how many padding tokens we need
            padding_needed = self.max_sentence_length - current_length
            # Add padding tokens to reach max_sentence_length
            paddings = [self.pad_token] * padding_needed
            token_embeddings.extend(paddings)
        # If the sentence is longer than max_sentence_length, truncate
        elif current_length > self.max_sentence_length:
            token_embeddings = token_embeddings[:self.max_sentence_length]
        return np.array(token_embeddings)

    def sentence2vec(self, sentence):
        token_embeddings = []
        for token in preprocess_sentence(sentence):
            if token in self.model:
                token_embeddings.append(self.model[token])

        if not token_embeddings:
            # Return a vector of zeros if no tokens were found
            return np.zeros(200)

        token_embeddings = np.array(token_embeddings)
        avg_embedding = np.mean(token_embeddings, axis=0)
        return avg_embedding

    def loinc2label(self, code):
        label = self.loinc[self.loinc["LOINC_NUM"] == code]["COMPONENT"]
        return label.iloc[0] if not label.empty else "None"

    def process(self, target, query):
        if (isinstance(target, str) and
                re.fullmatch(r'^\d{5,7}(-\d{1,2})?$', target)):
            target = self.loinc2label(target)

        target_embedding = self.sentence2vec(target)
        query_embedding = self.sentence2vecs(query)

        target_embedding = target_embedding.reshape(1, -1)

        stacked_embeddings = np.vstack((target_embedding, query_embedding))

        return stacked_embeddings

def build_model(loinc_vocab):
    query_seq_len = 100  # Maximum length for query tokens
    query_embedding_layer = build_query_embedding(max_sequence_length=query_seq_len)
    loinc_embedding_layer = build_loinc_embedding(loinc_vocab)

    # Model parameters
    loinc_seq_len = 1     # Maximum length for LOINC tokens
    dropout_rate = 0.1
    num_encoder_blocks = 2  # Number of Transformer blocks to stack for each branch

    # Define inputs for query and LOINC tokens
    query_input = tf.keras.Sequential(tf.keras.Input(shape=(query_seq_len,), name='query_input'),
                                      query_embedding_layer)
    loinc_input =  tf.keras.Sequential(tf.keras.Input(shape=(loinc_seq_len,), name='loinc_input'),
                                       loinc_embedding_layer)

    # Function to process an input sequence with a stack of TransformerEncoderBlock layers
    def process_sequence(x, num_blocks,
                         num_attention_heads = 8,
                         inner_dim = 512,
                         inner_activation = 'relu',
                    ):
        for _ in range(num_blocks):
            x = tfm.nlp.layers.TransformerEncoderBlock(
                num_attention_heads=num_attention_heads,
                inner_dim=inner_dim,
                inner_activation=inner_activation,
                output_dropout=dropout_rate,
                attention_dropout=dropout_rate,
                inner_dropout=dropout_rate
            )(x)
        # Pool over the sequence dimension to get a fixed-length vector representation
        return tf.keras.layers.GlobalAveragePooling1D()(x)

    # Process each branch
    query_encoded = process_sequence(query_input, num_encoder_blocks)
    loinc_encoded = process_sequence(loinc_input, num_encoder_blocks)

    # Combine the two representations.
    # Here we concatenate the vectors along with their element-wise absolute difference.
    combined = tf.keras.layers.Concatenate()([
        query_encoded,
        loinc_encoded,
        tf.abs(query_encoded - loinc_encoded)
    ])

    # Feed the combined representation through a small classifier head.
    x = tf.keras.layers.Dense(64, activation='relu')(combined)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Output probability between 0 and 1

    # Create and compile the model.
    model = tf.keras.Model(inputs=[query_input, loinc_input], outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Save model before returning
    model_save_dir = './model'
    model_name = "model.keras"
    os.makedirs(model_save_dir, exist_ok=True)
    model.save(os.path.join(model_save_dir, model_name))
    print(f"Model saved to {model_save_dir}")

    return model

def build_model_without_embedding():
    query_seq_len = 100  # Maximum length for query tokens
    # Model parameters
    num_encoder_blocks = 2
    dropout_rate = 0.1

    def transformer_encoder_block(x, num_heads, inner_dim, dropout_rate):
        # Self-attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=x.shape[-1] // num_heads
        )(x, x)
        attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward network
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(inner_dim, activation='relu'),
            tf.keras.layers.Dense(x.shape[-1])
        ])
        ffn_output = ffn(x)
        ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        return x
    # Function to process an input sequence with a stack of TransformerEncoderBlock layers
    def process_sequence(x, num_blocks, num_attention_heads=8, inner_dim=512, dropout_rate=0.1):
        for _ in range(num_blocks):
            x = transformer_encoder_block(
                x,
                num_heads=num_attention_heads,
                inner_dim=inner_dim,
                dropout_rate=dropout_rate
            )
        return tf.keras.layers.GlobalAveragePooling1D()(x)

    # Combine the two representations.
    # Here we concatenate the vectors along with their element-wise absolute difference.

    input = tf.keras.Input(shape=(query_seq_len + 1, 200), name='input')
    processed = process_sequence(input, num_encoder_blocks)
    # Feed the combined representation through a small classifier head.
    x = tf.keras.layers.Dense(64, activation='relu')(processed)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)  # Output probability between 0 and 1

    # Create and compile the model.
    model = tf.keras.Model(inputs=input, outputs=outputs)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Save model before returning
    model_save_dir = './model'
    model_name = "model.keras"
    os.makedirs(model_save_dir, exist_ok=True)
    model.save(os.path.join(model_save_dir, model_name))
    print(f"Model saved to {model_save_dir}")

    return model


def create_dataset(df, embedding, batch_size=32):
    # Create tensors from dataframe columns
    query_tensor = tf.convert_to_tensor(df['QUERY'].values, dtype=tf.string)
    target_tensor = tf.convert_to_tensor(df['TARGET'].values, dtype=tf.string)

    # Convert relevance labels: -1 → 0, 0 → 1, 1 → 2
    relevance_map = {-1: 0, 0: 1, 1: 2}
    labels = tf.convert_to_tensor([relevance_map[label] for label in df['RELEVANCE'].values],
                                  dtype=tf.int32)

    # Create initial dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((
        target_tensor, query_tensor, labels
    ))

    # Process function that works with raw tensors instead of dict
    def process_embedding(target, query, label):
        # Convert TF tensors to numpy for the embedding function
        target_np = target.numpy().decode('utf-8') if isinstance(target.numpy(), bytes) else target.numpy()
        query_np = query.numpy().decode('utf-8') if isinstance(query.numpy(), bytes) else query.numpy()

        # Process through embedding (which expects numpy)
        processed = embedding.process(target_np, query_np)

        # Return processed embedding and label
        return processed, label

    # TF wrapper function
    def tf_process_embedding(target, query, label):
        return tf.py_function(
            func=process_embedding,
            inp=[target, query, label],
            Tout=[tf.float32, tf.int32]
        )

    # Apply transformation using map
    dataset = dataset.map(
        tf_process_embedding,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return (dataset
            .shuffle(buffer_size=len(df))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))


def create_dataset_generator(df, embedding, batch_size=32):
    """
    Creates a generator that yields batches of data for model.fit
    using NumPy arrays until the final conversion to TensorFlow tensors.

    Args:
        df: Pandas DataFrame with 'QUERY', 'TARGET', and 'RELEVANCE' columns
        embedding: Embedding function that processes text
        batch_size: Size of batches to yield

    Yields:
        Tuple of (features, labels) as TensorFlow tensors
    """
    # Extract data from dataframe
    queries = df['QUERY'].values
    targets = df['TARGET'].values

    # Convert relevance labels: -1 → 0, 0 → 1, 1 → 2
    relevance_map = {-1: 0, 0: 1, 1: 2}
    labels = np.array([relevance_map[label] for label in df['RELEVANCE'].values],
                      dtype=np.int32)

    while True:
        # Create indices array for shuffling
        indices = np.arange(len(df))

        # Shuffle the indices
        np.random.shuffle(indices)

        # Process data in batches
        for start_idx in range(0, len(indices), batch_size):
            # Get indices for this batch
            batch_indices = indices[start_idx:start_idx + batch_size]

            # Initialize lists to store batch data
            batch_features = []
            batch_labels = []

            # Process each item in the batch
            for idx in batch_indices:
                target = targets[idx]
                query = queries[idx]
                label = labels[idx]

                # Decode bytes if necessary
                if isinstance(target, bytes):
                    target = target.decode('utf-8')
                if isinstance(query, bytes):
                    query = query.decode('utf-8')

                # Process through embedding
                processed = embedding.process(target, query)

                # Add to batch
                batch_features.append(processed)
                batch_labels.append(label)

            # Convert batch lists to numpy arrays
            batch_features_np = np.array(batch_features, dtype=np.float32)
            batch_labels_np = np.array(batch_labels, dtype=np.int32)

            # Convert numpy arrays to TensorFlow tensors just before yielding
            yield (tf.convert_to_tensor(batch_features_np),
                   tf.convert_to_tensor(batch_labels_np))


if __name__ == "__main__":
    import pandas as pd
    dataset_df = pd.read_csv(config.EXTRA_DF_PATH, sep=';')
    min_count = dataset_df['RELEVANCE'].value_counts().min()

    print("Distribution before balancing:")
    print(dataset_df['RELEVANCE'].value_counts())

    # Subsample each class to the minimum count to balance the dataset.
    dataset_df = dataset_df.groupby('RELEVANCE', group_keys=False).apply(
        lambda x: x.sample(min_count, random_state=42))


    loinc_df = pd.read_csv(config.LOINC_PATH)

    print("Dataset read")

    embedding = Embedding(debug=False, loinc=loinc_df)

    batch_size = 64
    test_split = 0.05
    train_sz = int(len(dataset_df) * (1-test_split))
    val_sz = int(len(dataset_df) * (test_split))
    val_batch = 2*batch_size

    steps_per_epoch = train_sz // batch_size
    if train_sz% batch_size != 0:
        steps_per_epoch += 1  # Add one more batch for the remainder

    # Create optimized dataset pipeline
    train_df, val_df = train_test_split(dataset_df, test_size=test_split, random_state=42, stratify=dataset_df['RELEVANCE'])

    # Create a ModelCheckpoint callback to save the best model
    checkpoint_path = "./model/best_model.keras"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',  # You can change this to 'val_loss' if you prefer
        verbose=1,
        save_best_only=True,
        mode='max'  # 'max' for accuracy, 'min' for loss
    )

    # Early stopping callback (optional)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Number of epochs with no improvement after which training will stop
        restore_best_weights=True,
        verbose=1
    )

    with tf.device('/CPU:0'):
        # Train the model
        model = build_model_without_embedding()
        model.fit(
            create_dataset_generator(train_df, embedding),
            validation_data=create_dataset_generator(val_df, embedding, batch_size=val_batch),
            epochs=40,
            steps_per_epoch = steps_per_epoch,
            validation_steps = val_sz //  val_batch,
            callbacks=[checkpoint_callback, early_stopping],
        )



