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
        sheet_names = [
            'glucose in blood',
            'bilirubin in plasma',
            'White blood cells count'
        ]
        self.sheet_name = sheet_names[2]

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
        df_filtered = parsed_df[['loinc_num', 'long_common_name']]
        return query, parsed_df, df_filtered


if __name__ == "__main__":
    DE = DataExtraction()
    query, parsed_df, df = DE.parsing_df()
