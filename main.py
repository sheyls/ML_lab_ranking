from data_extraction import DataExtraction
from filter_ranking import ranking

if __name__ == '__main__':
    DE = DataExtraction()
    query, parsed_df, df = DE.parsing_df()
    df_ranked = ranking(query, df)
    print(df_ranked)