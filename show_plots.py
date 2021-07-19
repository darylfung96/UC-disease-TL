import os
import matplotlib.pyplot as plt
import numpy as np
import torch

validation_terms = ['validation f1', 'validation precision', 'validation recall', 'validation loss']
plot_choices = ['GAIN', 'mice', 'mean']  # or None
plot_type = None

for validation_term in validation_terms:
    plt.clf()
    plt.ylabel(validation_term)
    plt.xlabel('epoch')
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
        current_result = result[validation_term]
        plt.plot(range(current_result.shape[0]), current_result, label=pth)
    plt.legend()
    os.makedirs(f'plots/plots results/{validation_term}', exist_ok=True)
    plt.savefig(f'plots/plots results/{validation_term}/{plot_type}.png', dpi=1200)
    # plt.show()


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
