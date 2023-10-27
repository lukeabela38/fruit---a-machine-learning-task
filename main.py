import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics

# Define list of Fruits
FRUITS = ["apple", "orange", "lemon"]

# Define the data points titles
VARIABLES = ["width", "weight", "height", "color_score"]

# Create a map of the fruit to its index 
FRUIT2LABEL = {
    "apple" : 1,
    "orange": 2,
    "lemon": 3
}

# Load in our CSV Data
def import_data(path_to_data):
    return pd.read_csv(path_to_data)

# Create a correlation matrix to help us interpret our data
def correlation_matrix(df_to_correlate):

    # generate correlation matrix
    df_to_correlate = df_to_correlate.drop(["fruit_name", "fruit_label"], axis=1)
    correlation_matrix = df_to_correlate.corr(method="pearson")

    # plot correlation matrix
    _, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, fmt='.4f', 
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")

    # save correlation matrix
    plt.savefig('artifacts/correlation_matrices/matrix.png', bbox_inches='tight', pad_inches=0.5)

# generate all our plots
def scatter_plot(df_to_plot):

    colour = "fruit_name"

    for i in range(len(VARIABLES)):
        for j in range(len(VARIABLES)):
            if VARIABLES[i] != VARIABLES[j]:
                x_axis = VARIABLES[i]
                y_axis = VARIABLES[j]

                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df_to_plot, x=x_axis, y=y_axis, hue=colour)
                plt.savefig(f"artifacts/scatter_plots/{x_axis}_vs_{y_axis}.png")

# use zscore to normalise our numeric data.
# https://www.geeksforgeeks.org/data-normalization-in-data-mining/
# We also shuffle our data (good practice)
def preprocess_data(df_to_process):

    for variable in VARIABLES:
        df_to_process[variable] = stats.zscore(df_to_process[variable])
    df_to_process = df_to_process.sample(frac = 1) # Shuffle
    return df_to_process

# split data into test set and train set
def split_data(df_to_split):
    train_df, test_df = train_test_split(df_to_split, test_size=0.2)
    
    y_train = train_df["fruit_label"].to_numpy()
    y_test = test_df["fruit_label"].to_numpy()

    x_train = train_df.drop(["fruit_name", "fruit_label"], axis=1).to_numpy()
    x_test = test_df.drop(["fruit_name", "fruit_label"], axis=1).to_numpy()

    return x_train, y_train, x_test, y_test

# train classifier
def binary_classifier_train(x_train, y_train, fruit_to_classify):

    # logic to ensure we only have 2 possible results - fruit (1) or not fruit (0)
    fruit_index = FRUIT2LABEL[fruit_to_classify]
    for i in range(len(y_train)):
        if y_train[i] != fruit_index:
            y_train[i] = 0
    y_train = y_train/np.max(y_train)

    clf = RandomForestClassifier()   
    clf.fit(x_train, y_train) 
    return clf

# test classifier
def binary_classifier_test(clf, x_test, y_test, fruit_to_classify):

    # logic to ensure we only have 2 possible results - fruit (1) or not fruit (0)
    fruit_index = FRUIT2LABEL[fruit_to_classify]
    for i in range(len(y_test)):
        if y_test[i] != fruit_index:
            y_test[i] = 0
    y_test = y_test/np.max(y_test)
    y_pred = clf.predict(x_test) 

    return y_pred

# results include confusion matrix
def binary_classifier_results(y_test, y_pred, fruit_to_classify):

    y_test = y_test/np.max(y_test)

    f1_score = round(metrics.f1_score(y_test, y_pred, average='macro'),4)
    accuracy = round(metrics.accuracy_score(y_test, y_pred), 4)

    plt.figure(figsize=(8, 6))
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()

    plt.savefig(f"artifacts/confusion_matrices/{fruit_to_classify}_{accuracy}_{f1_score}.png")

def multi_classifier_train(x_train, y_train):

    clf = RandomForestClassifier(n_estimators=1000)   
    clf.fit(x_train, y_train) 
    return clf

def multi_classifier_test(clf, x_test, y_test):

    y_test = y_test/np.max(y_test)
    y_pred = clf.predict(x_test) 

    return y_pred

def multi_classifier_results(y_test, y_pred):

    f1_score = round(metrics.f1_score(y_test, y_pred, average='macro'),4)
    accuracy = round(metrics.accuracy_score(y_test, y_pred), 4)

    plt.figure(figsize=(8, 6))
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Apple", "Orange", "Lemon"])
    cm_display.plot()

    plt.savefig(f"artifacts/confusion_matrices/multiclass_{accuracy}_{f1_score}.png")

def main() -> None:

    ## Task 1
    path_to_data = "data/fruitDataset.csv"
    df = import_data(path_to_data)

    correlation_matrix(df)
    scatter_plot(df)

    ## Task 2
    for fruit_to_classify in FRUITS:
        processed_df = preprocess_data(df)
        x_train, y_train, x_test, y_test = split_data(processed_df)

        clf = binary_classifier_train(x_train, y_train, fruit_to_classify)
        y_pred = binary_classifier_test(clf, x_test, y_test, fruit_to_classify)
        assert len(set(y_pred)) == 2 ## make sure we have exactly 2 outputs to satisfy binary classifier
        binary_classifier_results(y_test, y_pred, fruit_to_classify)

    ## Task 3
    x_train, y_train, x_test, y_test = split_data(processed_df)

    clf = multi_classifier_train(x_train, y_train)
    y_pred = multi_classifier_test(clf, x_test, y_test)
    assert len(set(y_pred)) == 3 ## make sure we have exactly 2 outputs to satisfy multi class classifier

    multi_classifier_results(y_test, y_pred)

    return 0

if __name__ == "__main__":
    main()