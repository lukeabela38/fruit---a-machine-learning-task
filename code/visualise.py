import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics

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
    plt.close()

# generate all our plots
def scatter_plot(df_to_plot, variables):

    for i in range(len(variables)):
        for j in range(len(variables)):
            if variables[i] != variables[j]:
                x_axis = variables[i]
                y_axis = variables[j]

                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df_to_plot, x=x_axis, y=y_axis, hue="fruit_name")
                plt.savefig(f"artifacts/scatter_plots/{x_axis}_vs_{y_axis}.png")
                plt.close()

def confusion_matrix(y_test, y_pred, label=None, display_labels = None):

    if display_labels == None:
        display_labels = [False, True]

    if len(display_labels) == 2:
        y_test = y_test/np.max(y_test)

    plt.figure(figsize=(8, 8))
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
    cm_display.plot()

    plt.savefig(f"artifacts/confusion_matrices/cm_{label}.png")
    plt.close()
