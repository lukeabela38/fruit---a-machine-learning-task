import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

# Load in our CSV Data
def import_data(path_to_data):
    return pd.read_csv(path_to_data)

# use zscore to normalise our numeric data.
# https://www.geeksforgeeks.org/data-normalization-in-data-mining/
# We also shuffle our data (good practice)
def preprocess_data(df, variables):
    for variable in variables:
        df[variable] = stats.zscore(df[variable])
    return df.sample(frac = 1) # Shuffle

# split data into test set and train set
def split_data(df_to_split):
    train_df, test_df = train_test_split(df_to_split, test_size=0.2)
    
    y_train = train_df["fruit_label"].to_numpy()
    y_test = test_df["fruit_label"].to_numpy()

    x_train = train_df.drop(["fruit_name", "fruit_label"], axis=1).to_numpy()
    x_test = test_df.drop(["fruit_name", "fruit_label"], axis=1).to_numpy()

    return x_train, y_train, x_test, y_test

def preprocess_binarisation_data(y: list, fruit_to_classify: int):
    
    # logic to ensure we only have 2 possible results - fruit (1) or not fruit (0)
    assert (fruit_to_classify) > 0
    for i in range(len(y)):
        if y[i] != fruit_to_classify:
            y[i] = 0
    return y