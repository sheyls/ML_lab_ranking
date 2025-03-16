import config
from data_extraction import DataExtraction
from filter_ranking import ranking
from model import *
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score


def train(dataset_df=pd.read_csv(config.EXTRA_DF_PATH, sep=';'), loinc_df = pd.read_csv(config.LOINC_PATH), batch_size=64, epochs=1):
    """

    :param dataset_df: It has to be in shape TARGET (valid loinc code), QUERY (a sentence string), RELEVANCE (an integer from -1, 0, 1)
    :return:
    """

    embedding = Embedding(debug=False, loinc=loinc_df)

    test_split = 0.05
    train_sz = int(len(dataset_df) * (1 - test_split))
    val_sz = int(len(dataset_df) * (test_split))
    val_batch = 2 * batch_size

    steps_per_epoch = train_sz // batch_size
    if train_sz % batch_size != 0:
        steps_per_epoch += 1  # Add one more batch for the remainder

    # Create optimized dataset pipeline
    train_df, val_df = train_test_split(dataset_df, test_size=test_split, random_state=42,
                                        stratify=dataset_df['RELEVANCE'])

    # Create a ModelCheckpoint callback to save the best model
    checkpoint_path = "./model/best_model.keras"

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

    with tf.device('/CPU:0'): # Remove as needed to train with the GPU
        # Train the model
        model = build_model_without_embedding()
        model.fit(
            create_dataset_generator(train_df, embedding),
            validation_data=create_dataset_generator(val_df, embedding, batch_size=val_batch),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_sz // val_batch,
            callbacks=[checkpoint_callback, early_stopping],
        )


def validate(model_path, dataset_df=pd.read_csv(config.EXTRA_DF_PATH, sep=';'), loinc_df = pd.read_csv(config.LOINC_PATH), batch_size=64):
    """

    :param dataset_df: It has to be in shape TARGET (valid loinc code), QUERY (a sentence string), RELEVANCE (an integer from -1, 0, 1)
    :return:
    """

    embedding = Embedding(debug=False, loinc=loinc_df)
    data_sz = int(len(dataset_df))
    val_gen = create_dataset_generator(dataset_df, embedding)

    steps = data_sz // batch_size
    if data_sz % batch_size != 0:
        steps += 1  # Add one more batch for the remainder

    with tf.device('/CPU:0'): # Remove as needed to train with the GPU
        # Train the model
        model = load_model(model_path)
        # Initialize lists to store predictions and true labels.
        all_preds = []
        all_labels = []

        # Loop over the validation dataset for the given number of steps.
        for _ in range(steps):
            X_batch, y_batch = next(val_gen)
            # Get model predictions for the current batch.
            preds = model.predict(X_batch)

            # Assuming a multi-class classification problem where predictions are probabilities,
            # take the argmax to get predicted class labels.
            pred_labels = np.argmax(preds, axis=1)

            # Handle true labels:
            # If y_batch is one-hot encoded, convert to label indices.
            if y_batch.ndim == 2:
                true_labels = np.argmax(y_batch, axis=1)
            else:
                true_labels = y_batch

            all_preds.extend(pred_labels)
            all_labels.extend(true_labels)

        # Compute overall accuracy.
        accuracy = accuracy_score(all_labels, all_preds)

        # Compute macro-average F1 score (averaging F1 over all classes equally).
        f1_macro = f1_score(all_labels, all_preds, average='macro')

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 Score (Macro): {f1_macro:.4f}")

if __name__ == '__main__':
    # LOAD DATA
    loinc_df = pd.read_csv(config.LOINC_PATH)
    print("LOINC read")

    DE = DataExtraction()
    dataset_df = None
    for sheet in DE.sheet_names:
        DE.sheet_name = sheet
        query, parsed_df, df = DE.parsing_df()

        # FORMATTING CHANGE OF THE DF FOR COMPATIBILITY
        df_ranked = ranking(query, df)
        dataset_df_ = df_ranked.copy()
        dataset_df_["QUERY"] = query
        dataset_df_.rename(columns={"score": "RELEVANCE"}, inplace=True)
        dataset_df_["RELEVANCE"] = dataset_df_["RELEVANCE"] - 2

        # loinc_df["LOINC_NUM"] of loinc_df where loinc_df["COMPONENT"] == dataset_df["long_common_name"]
        # component_to_loinc = loinc_df.groupby("LONG_COMMON_NAME")["LOINC_NUM"].first()
        dataset_df_["TARGET"] = dataset_df_["long_common_name"]

        if dataset_df is None:
            dataset_df = dataset_df_
        else:
            dataset_df = pd.concat([dataset_df, dataset_df_])
        print(dataset_df.shape)
    print(dataset_df.head())



    # TRAIN or VALIDATE
    train(dataset_df, loinc_df)
    # validate(model_path="./model/trained_model.keras", dataset_df=dataset_df, loinc_df=loinc_df)

