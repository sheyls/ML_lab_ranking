import pandas as pd

from data_extraction import DataExtraction
from filter_ranking import ranking

if __name__ == '__main__':
    sheet_names = [
        'glucose in blood',
        'bilirubin in plasma',
        'White blood cells count',
        'Creatinine in blood',
        'Cholesterol In Plasma'
    ]
    with pd.ExcelWriter('dataset/ranked_df.xlsx') as writer:
        for sheet in sheet_names:
            DE = DataExtraction(sheet)
            query, parsed_df, df = DE.parsing_df()
            df_ranked = ranking(query, df)
            df_ranked.to_excel(
                writer,
                sheet_name=sheet,
                index=False
            )
