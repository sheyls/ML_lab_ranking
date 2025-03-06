import pandas as pd
from config import DATA_DIR
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Download the punkt tokenizer if not already downloaded
nltk.download('punkt', dir="C:\\nltk_data")
nltk.download('punkt_tab', dir="C:\\nltk_data")

raw_data = pd.DataFrame(columns=["Query", "Keyword", "Target"])
for file in os.walk(DATA_DIR):
    excel_data = pd.read_excel(os.path.join(DATA_DIR, file), sheet_name=None)

    # The keys of the dictionary are the sheet names.
    sheet_names = list(excel_data.keys())
    print("Sheet names:", sheet_names)

    # Optionally, if you want to inspect the contents of each sheet:
    for name, df in excel_data.items():
        DATA = pd.concat(DATA, df[["Query", "Keyword", "Target"]])


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = raw_data.copy()

# Initialize the stop words, stemmer, and lemmatizer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def process_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)
    # Remove stop words (comparison in lowercase)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    # Apply stemming to the filtered tokens
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    # Apply lemmatization to the filtered tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return tokens, filtered_tokens, stemmed_tokens, lemmatized_tokens

# Process each column and create new columns for the results
df[['query_tokens', 'query_filtered', 'query_stemmed', 'query_lemmatized']] = df['query'].apply(lambda x: pd.Series(process_text(x)))
df[['keyword_tokens', 'keyword_filtered', 'keyword_stemmed', 'keyword_lemmatized']] = df['keyword'].apply(lambda x: pd.Series(process_text(x)))
