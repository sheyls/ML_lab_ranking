import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

# Use STOP_WORDS provided by spaCy as stop_words
stop_words = STOP_WORDS


def ranking(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe based on the query and return the modified dataframe with a ternary ranking system:
    - If the query is in the text, match almost completely -> score = 1
    - If the query is partially in the text -> score = 2
    - If the query is not in the text, no match -> score = 3

    :param query: The query to filter the dataframe
    :param df: The dataframe to filter
    :return: A dataframe with the column 'score' added, sorted by score
    """

    def compute_score(text):
        # Standardize text and query
        text = text.lower()
        q = query.lower()  # use a local variable to avoid modifying the outer query
        text = text.replace(',', ' ')
        # Remove stopwords using spaCy's list
        text = ' '.join([word for word in text.split() if word not in stop_words])
        
        text_words = set(text.split())
        query_words = set(q.split())
        
        # Exact match: all words from the query are present in the text (regardless of order or position)
        if query_words.issubset(text_words):
            return 1
        # Partial match: at least one word from the query is in the text
        elif query_words.intersection(text_words):
            return 2
        # No match: none of the words are present
        else:
            return 3
    
    df['score'] = df['text'].apply(compute_score)
    df = df.sort_values(by='score')
    return df
