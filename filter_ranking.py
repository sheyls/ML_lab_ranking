import re

import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

# Use STOP_WORDS provided by spaCy as stop_words
stop_words = STOP_WORDS


def ranking(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe based on the query and return the modified dataframe with
    a ternary ranking system:
    - If the query is in the text, match almost completely -> score = 1
    - If the query is partially in the text -> score = 2
    - If the query is not in the text, no match -> score = 3

    :param query: The query to filter the dataframe
    :param df: The dataframe to filter
    :return: A dataframe with the column 'score' added, sorted by score
    """

    def clean_text(text):
        # Eliminar caracteres especiales y puntuación (manteniendo solo letras y espacios)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    def compute_score(text):
        # Standardize text and query
        text = text.lower()
        q = query.lower()  # use a local variable to avoid modifying the outer query
        text = text.replace(',', ' ')
        # Remove stopwords using spaCy's list
        q = clean_text(q)
        text = clean_text(text)

        text_words = set(text.split())
        query_words = set(q.split())

        # Exact match: all words from the query are present in the text
        # (regardless of order or position)
        if query_words.issubset(text_words):
            return 1
        # Partial match: at least one word from the query is in the text
        elif query_words.intersection(text_words):
            return 2
        # No match: none of the words are present
        else:
            return 3

    df['score'] = df['long_common_name'].apply(compute_score)
    df = df.sort_values(by='score')
    return df
