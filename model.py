import gensim
import numpy as np
import tensorflow as tf

max_sequence_length = 100  # for example
word_index = {}

bio_wordvec = gensim.models.KeyedVectors.load_word2vec_format(
    'BioWordVec_PubMed_MIMICIII_d200.bin', binary=True
)

embedding_dim = bio_wordvec.vector_size
vocab_size = len(word_index) + 1  # +1 to account for padding token (index 0)

# Create the embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if word in bio_wordvec:
        embedding_matrix[i] = bio_wordvec[word]
    else:
        # For words not in BioWordVec, you can either leave the row as zeros
        # or initialize it randomly. Here, we use a random normal initializer.
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

# Now, create a Keras Embedding layer with these weights.
# Set trainable=True if you wish to fine-tune the embeddings during training.
query_embedding_layer = tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_sequence_length,
    trainable=True
)

embedding_matrix = embedding_matrix.copy()
for word, i in word_index.items():
    if word in bio_wordvec:
        embedding_matrix[i] = bio_wordvec[word]
    else:
        # For words not in BioWordVec, you can either leave the row as zeros
        # or initialize it randomly. Here, we use a random normal initializer.
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

# Now, create a Keras Embedding layer with these weights.
# Set trainable=True if you wish to fine-tune the embeddings during training.
vocabulary_embedding_layer = tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_sequence_length,
    trainable=True
)

from tfm.nlp.layers import TransformerEncoderBlock  # make sure to install tf-models-official

# Model parameters
query_seq_len = 50     # Maximum length for query tokens
loinc_seq_len = 10     # Maximum length for LOINC tokens
embedding_dim = 128    # Dimensionality of your pre-computed embeddings
num_attention_heads = 8
inner_dim = 512
inner_activation = 'relu'
dropout_rate = 0.1
num_encoder_blocks = 2  # Number of Transformer blocks to stack for each branch

# Define inputs for query and LOINC tokens
query_input = tf.keras.Input(shape=(query_seq_len, embedding_dim), name='query_input')
loinc_input = tf.keras.Input(shape=(loinc_seq_len, embedding_dim), name='loinc_input')

# Function to process an input sequence with a stack of TransformerEncoderBlock layers
def process_sequence(x, num_blocks):
    for _ in range(num_blocks):
        x = TransformerEncoderBlock(
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