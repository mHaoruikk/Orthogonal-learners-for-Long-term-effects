import matplotlib.pyplot as plt
import numpy as np


def plot_cate_predictions(X:np.ndarray, true_cate:np.ndarray, pred_cate:np.ndarray, 
                          groupby = "X1+X2+X3+X4",
                          title:str="CATE Predictions"):
    """
    Plots true vs predicted CATE values against the aggregated pre-treatment covariates specified by groupby.

    Parameters:
    - X: Covariate matrix (n_samples x n_features)
    - true_cate: True CATE values
    - pred_cate: Predicted CATE values
    - title: Title of the plot
    """
    plt.figure(figsize=(8, 6))
    if groupby == "X1+X2+X3+X4":
        agg_X = X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3]
    else:
        raise ValueError("Unsupported groupby option")
    
    plt.plot(agg_X, true_cate, 'o', label='True CATE', alpha=0.6, color = 'orange')
    plt.plot(agg_X, pred_cate, 'x', label='Predicted CATE', alpha=0.6, color = 'blue')
    plt.xlabel('Covariates')
    plt.ylabel('CATE Values')
    plt.title(title)
    plt.legend()
    plt.show()