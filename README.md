# A self-knowledge distillation-driven CNN-LSTM model for predicting disease outcomes using longitudinal microbiome data 


### Getting Started
Make sure you pip install the required packages.

To run training
```
python main.py
```

The training would test out different models including the permutation of:
- CNNLSTM/LSTM
- PCA/non-PCA
- pad/pad in sequence
- gradual unfreezing
- concat poolings
- self-distillation (first and second)
- imputation (GAIN, mean, mice)


After training, plots will be save in the directory of:
```
plots/aveage F1 plots/plots for ...
```

To see the plots, run
```
python show_plots.py
```






