import pandas as pd

def dataset_analysis(df, values):

    analysis = {}
    for value in values:
        filtered_df = df[df['fruit_name'] == value]
        analysis[value] = filtered_df.shape[0]
    return analysis
