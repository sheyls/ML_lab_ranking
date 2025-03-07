from pathlib import Path

import pandas as pd


class DataExtraction():
    def __init__(self) -> None:
        self.path = Path.cwd()
        sheet_names = [
            'glucose in blood',
            'bilirubin in plasma',
            'White blood cells count'
        ]
        self.sheet_name = sheet_names[2]

    def extracting_query(self, df):
        query = df.iat[0, 0].replace("Query: ", "")
        return query

    def extracting_df(self, df):
        headers = df.iloc[2]
        parsed_df = df[3:].copy()
        parsed_df.columns = headers
        parsed_df.reset_index(drop=True, inplace=True)
        return parsed_df

    def parsing_df(self):
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
