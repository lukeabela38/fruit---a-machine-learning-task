import numpy as np
from code.constants import Fruit, Data, TrainingAlgorithms
from code.utils import import_data, split_data, preprocess_data, preprocess_binarisation_data
from code.visualise import correlation_matrix, scatter_plot, confusion_matrix, roc_curve
from code.classifier import MLTrainer
from code.analysis import dataset_analysis

def main() -> int:

    ## Control (vary from 0 -> 3)
    algorithm_selection: int = 1

    ## Task 1
    path_to_data: str = "data/fruitDataset.csv"
    df = import_data(path_to_data)

    analysis: dict = dataset_analysis(df, list(Fruit.fruit_index.value.keys()))

    correlation_matrix(df)
    scatter_plot(df, Data.variables.value)
3
    ## Task 2
    for fruit_to_classify in list(Fruit.fruit_index.value.values()):
        processed_df = preprocess_data(df, Data.variables.value)
        x_train, y_train, x_test, y_test = split_data(processed_df)

        y_train = preprocess_binarisation_data(y_train, fruit_to_classify)
        y_test = preprocess_binarisation_data(y_test, fruit_to_classify)

        fruit_classifier = MLTrainer(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            training_algorithm=TrainingAlgorithms.algorithms.value[algorithm_selection],
            normalise=True,
            )
        
        y_pred = fruit_classifier.get_classifier_pred()
        binary_classifier_results = fruit_classifier.get_classifier_results()

        confusion_matrix(y_test=y_test, y_pred=y_pred, label=fruit_to_classify)
        roc_curve(y_test, y_pred, label=fruit_to_classify)
        print(binary_classifier_results)

    ## Task 3
    x_train, y_train, x_test, y_test = split_data(processed_df)
    fruit_classifier = MLTrainer(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        training_algorithm=TrainingAlgorithms.algorithms.value[algorithm_selection],
        normalise=False,
        )
    
    y_pred = np.array(fruit_classifier.get_classifier_pred())
    multi_classifier_results = fruit_classifier.get_classifier_results()

    display_labels = list(Fruit.fruit_index.value.values())
    confusion_matrix(y_test=y_test, y_pred=y_pred, label="multiclass", display_labels=display_labels)

    print(multi_classifier_results)

    return 0

if __name__ == "__main__":
    main()
