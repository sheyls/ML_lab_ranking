from pathlib import Path

import pandas as pd


class DataExtraction():
    def __init__(self) -> None:
        """
        Initializes the DataExtraction class.

        - Sets the default path to the current working directory.
        - Defines a list of sheet names.
        - Selects 'White blood cells count' as the default sheet name.
        """
        self.path = Path.cwd()
        self.sheet_names = [
            'glucose in blood',
            'bilirubin in plasma',
            'White blood cells count',
            'Creatinine in blood',
            'Cholesterol In Plasma'
        ]
        self.sheet_name = self.sheet_names[0]

    def extracting_query(self, df):
        """
        Extracts the query string from the first cell of the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            str: The extracted query string with 'Query: ' removed.
        """
        query = df.iat[0, 0].replace("Query: ", "")
        return query

    def extracting_df(self, df):
        """
        Extracts and formats the main dataset from the given DataFrame.

        - Uses the third row as column headers.
        - Extracts rows starting from the fourth row.
        - Resets the index.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A parsed DataFrame with correctly assigned headers.
        """
        headers = df.iloc[2]
        parsed_df = df[3:].copy()
        parsed_df.columns = headers
        parsed_df.reset_index(drop=True, inplace=True)
        return parsed_df

    def parsing_df(self):
        """
        Reads and processes the dataset from an Excel file.

        - Reads an Excel sheet into a DataFrame with no headers.
        - Extracts the query string.
        - Processes and formats the DataFrame.
        - Filters specific columns: 'loinc_num' and 'long_common_name'.

        Returns:
            tuple: (query string, parsed DataFrame, filtered DataFrame)
        """
        df = pd.read_excel(
            self.path / 'dataset/Loinc Dataset v2.xlsx', sheet_name=self.sheet_name, header=None
        )
        query = self.extracting_query(df)
        parsed_df = self.extracting_df(df)
        df_filtered = parsed_df[['long_common_name']]
        return query, parsed_df, df_filtered


def testing_dataset_with_jaccard():
    import pandas as pd
    from sklearn.metrics import jaccard_score
    from sklearn.feature_extraction.text import CountVectorizer
    from tqdm import tqdm
    loinc_path = "./dataset/LoincTableCore/LoincTableCore.csv"
    df = pd.read_csv(loinc_path)
    df_useful = df[["LOINC_NUM", "COMPONENT"]]
    df_useful.rename(columns={"LOINC_NUM": "TARGET", "COMPONENT": "QUERY"}, inplace=True)

    # Convert queries to a binary bag-of-words matrix
    vectorizer = CountVectorizer(binary=True)
    query_matrix = vectorizer.fit_transform(df_useful["QUERY"])

    # Set a threshold for 'close to zero'
    bad_threshold = 0.01  # Adjust this value if necessary
    mediocre_interval = (0.4, 0.7)

    # Compute Jaccard similarity for each pair and filter pairs close to zero
    pairs_filtered = []
    bad_cases = 0
    mediocre_cases = 0
    progress_bar = tqdm(total=df_useful.shape[0], desc="Processing pairs")
    n_pairs = len(df_useful)  # Set the number of random pairs you want
    sz = len(df_useful)
    indices = np.random.uniform(size=(n_pairs, 2))

    for k in range(n_pairs):
        pair = np.round(indices[k] * sz).astype(int)
        i, j = pair
        # Convert the sparse vectors to dense arrays for computation
        vec_i = query_matrix[i].toarray()[0]
        vec_j = query_matrix[j].toarray()[0]
        jac_sim = jaccard_score(vec_i, vec_j)

        if jac_sim <= bad_threshold:
            a, b = df_useful.iloc[i]["QUERY"], df_useful.iloc[j]["QUERY"]
            pairs_filtered.append((df_useful.iloc[i]["TARGET"], df_useful.iloc[j]["QUERY"], -1))
            bad_cases += 1
            progress_bar.update(1)  # Manually update the tqdm bar
        elif mediocre_interval[0] < jac_sim < mediocre_interval[1]:
            a, b = df_useful.iloc[i]["QUERY"], df_useful.iloc[j]["QUERY"]
            pairs_filtered.append((df_useful.iloc[i]["TARGET"], df_useful.iloc[j]["QUERY"], 0))
            mediocre_cases += 1
            progress_bar.update(1)  # Manually update the tqdm bar

        if bad_cases + mediocre_cases >= df_useful.shape[0]:
            break

    # Convert the filtered results to a DataFrame
    df_useful['RELEVANCE'] = 1
    extra_queries = pd.DataFrame(pairs_filtered, columns=["TARGET", "QUERY", "RELEVANCE"])

    dataset = pd.concat([df_useful, extra_queries], ignore_index=True)
    dataset.to_csv("./extra_queries.csv", index=False, sep=";")
    print(extra_queries)

if __name__ == "__main__":
    DE = DataExtraction()
    query, parsed_df, df = DE.parsing_df()
