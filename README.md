# Fruit

## Task 1

- Load Data 
- Correlation Matrix (https://www.geeksforgeeks.org/create-a-correlation-matrix-using-python/)
- Scatter Plots (https://www.geeksforgeeks.org/plotting-correlation-matrix-using-python/)

## Task 2 

- Binary Classifier (Random Forest) (https://www.geeksforgeeks.org/getting-started-with-classification/)
- Train/Test Split (80/20) (https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/)
- F1 Score (https://www.statology.org/f1-score-in-python/) + Confusion Matrix

## Task 3

- Multiclass Classifier (https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/)

# What does the code do?

Importing Libraries: The code starts by importing various Python libraries that are necessary for data analysis, visualization, and machine learning. These libraries include Pandas, Seaborn, Matplotlib, NumPy, SciPy, Scikit-Learn, and more.

### Defining Constants:

FRUITS: A list of fruit names, including "apple," "orange," and "lemon."
VARIABLES: A list of data point titles, such as "width," "weight," "height," and "color_score."
FRUIT2LABEL: A dictionary that maps fruit names to numeric labels (e.g., "apple" maps to 1).
Data Import:

import_data(path_to_data): A function to read data from a CSV file specified by path_to_data using Pandas. It returns a DataFrame containing the data.
Correlation Matrix:

correlation_matrix(df_to_correlate): A function that generates a correlation matrix for the data in the DataFrame df_to_correlate. It uses Seaborn to create a heatmap of the correlations and saves the plot as an image.
Scatter Plots:

scatter_plot(df_to_plot): A function that creates scatter plots for different variable pairs in the data. It iterates through pairs of variables and uses Seaborn to create scatter plots, saving each plot as an image.
Data Preprocessing:

preprocess_data(df_to_process): A function that standardizes the numeric data using the z-score normalization technique and shuffles the data by randomly sampling it. It returns the preprocessed DataFrame.
Data Splitting:

split_data(df_to_split): A function that splits the data into training and testing sets. It separates features (X) and labels (Y) for both sets and returns them.
Binary Classifier Training and Testing:

binary_classifier_train(x_train, y_train, fruit_to_classify): A function that trains a binary classifier for a specific fruit. It sets up binary classification, where one fruit is the positive class (1) and all other fruits are the negative class (0).
binary_classifier_test(clf, x_test, y_test, fruit_to_classify): A function that tests the binary classifier on the test data and returns predictions.
binary_classifier_results(y_test, y_pred, fruit_to_classify): A function that evaluates the binary classifier's results, calculates the F1 score, and creates a confusion matrix plot.
Multi-Class Classifier Training and Testing:

multi_classifier_train(x_train, y_train): A function that trains a multi-class classifier using a random forest classifier with 1000 estimators.
multi_classifier_test(clf, x_test, y_test): A function that tests the multi-class classifier on the test data and returns predictions.
multi_classifier_results(y_test, y_pred): A function that evaluates the multi-class classifier's results, calculates the F1 score, and creates a confusion matrix plot.
Main Function:

main(): The main function that orchestrates the execution of all tasks. It loads data, performs correlation matrix and scatter plot analysis, trains and tests binary classifiers for each fruit, and trains and tests a multi-class classifier.
Execution:
The script concludes with an if __name__ == "__main__": block that ensures the main function is executed when the script is run as the main program.
In summary, this code performs data analysis and classification tasks on a dataset of fruits. It includes data preprocessing, visualization, and the training and testing of both binary and multi-class classifiers. The results are saved as images in specific directories.

## FAQs

### How to Run the Code

```bash
python main.py
```

Alternatively try the notebook!

### Where to See Results

You can go over to the artifacts folder, where you will find scatter plots (the naming convention is self explanatory), a correlation matrix (not required for this task), and confuison matrices. The confusion matrices follow the naming pattern:

```
<classifier>_<accuracy_score>_<f1_score>.png
```

The classifier may be a binary classifier trained for apple, lemon, or orange, or a multiclass classifier. 

You need accuracy, so feel free to ignore F1 score.

### Classification

We used a random forest classifier. You can definitely use others, and there are definitely better ones available.

### MISC

Why do the results vary everytime I run the code? Good Question! Everytime we do a train test split, just prior we are shuffling our data. That means, that every time we have a different version of train data and test data, so our results will vary (somewhat). If you want more definitive results, why not execute multiple times and take averages.

