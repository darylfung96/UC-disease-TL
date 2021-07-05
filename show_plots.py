import os
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import torch

validation_terms = ['validation f1', 'validation precision', 'validation recall', 'validation loss']

for validation_term in validation_terms:
    plt.clf()
    plt.ylabel(validation_term)
    plt.xlabel('epoch')
    for pth in os.listdir('plots/average F1 plots/'):
        result = torch.load(f'plots/average F1 plots/{pth}')
        current_result = result[validation_term]
        plt.plot(range(current_result.shape[0]), current_result, label=pth)
    plt.legend()
    plt.show()


def plot_confusion_matrix(confusion_matrix, name=""):
    fig, ax = plt.subplots()
    ax.matshow(confusion_matrix[:-2, :-2], cmap=plt.cm.Blues)
    for i in range(confusion_matrix.shape[0]-2):
        for j in range(confusion_matrix.shape[1]-2):
            ax.text(i, j, str(int(confusion_matrix[i][j])), va='center', ha='center')
    plt.title(f'confusion matrix({name})')
    plt.show()


# get the confusion matrix for each model
for pth in os.listdir('plots/average F1 plots/'):
    if pth != "LSTM PADDED":
        continue
    result = torch.load(f'plots/average F1 plots/{pth}')
    confusion_matrix = result['validation confusion matrix']
    plot_confusion_matrix(confusion_matrix, name=pth)
