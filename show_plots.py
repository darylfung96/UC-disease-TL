import os
import matplotlib.pyplot as plt
from itertools import chain
import pandas as pd
import numpy as np
import torch

validation_terms = ['validation f1', 'validation precision', 'validation recall', 'validation loss', 'validation auc']
columns = ['model']
for validation_term in validation_terms:
    columns += [f'{validation_term} mean']
    columns += [f'{validation_term} std']
all_rows = []

plot_choices = ['GAIN', 'mice', 'mean']  # or None
plot_type = 'mean'

# plot as table
for pth in os.listdir('plots/average F1 plots/'):

    # plot those without imputations
    if plot_type is None:
        should_continue = False
        for plot_choice in plot_choices:
            if plot_choice in pth:
                should_continue = True
                break
        if should_continue:
            continue
    else:
    # plot those with imputations
        if plot_type not in pth:
            continue

    result = torch.load(f'plots/average F1 plots/{pth}')
    mean = list(result['mean'].values())[:-1]
    std = list(result['std'].values())

    all_rows.append([pth.replace('plots for ', '')] + list(chain(*zip(mean, std))))
os.makedirs(f'plots/plots results/', exist_ok=True)
pd.DataFrame(all_rows, columns=columns).to_csv(f'plots/plots results/results.csv')


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


# get best f1 score sorted descending order
f1_dict = {}
for pth in os.listdir('plots/average F1 plots/'):
    result = torch.load(f'plots/average F1 plots/{pth}')
    validation_f1 = result['validation f1']
    best_validation_f1 = np.max(np.array(validation_f1))
    f1_dict[pth] = best_validation_f1
f1_data = list(f1_dict.items())
sorted_f1_data = sorted(f1_data, key=lambda l:l[1], reverse=True)
print(sorted_f1_data)
