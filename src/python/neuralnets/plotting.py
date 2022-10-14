"""
Utils for plotting
"""
import types
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pdb import set_trace

sns.set_style('whitegrid')
np.random.seed(1)


def plot_dataset(t, train, val, test, filename: Optional[str] = None, display: Optional[bool] = False):
    """Plot the whole MacKey Glass dataset"""
    n_train = len(train)
    n_val = len(val)
    n_test = len(test)

    # Setup index
    i_train = n_train
    i_val = i_train + n_val
    i_test = i_val + n_test

    fig, ax = plt.subplots(figsize=(11, 5))
    plt.plot(t[:i_train], train[:, 0], label="Train Data", linewidth=2.6)
    plt.plot(t[i_train:i_val], val[:, 0], label="Val Data", linewidth=2.6)
    plt.plot(t[i_val:i_test], test[:, 0], label="Test Data", linewidth=2.6)
    plt.axvline(t[0] + i_train, color='#333333', linestyle=':')
    plt.axvline(t[0] + i_val, color='#333333', linestyle=':')
    plt.legend()
    ax.set_title("Mackey Glass Series", fontsize=16, pad=12)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x(t)$")
    plt.setp(ax.spines.values(), color='#374151')

    if filename:
        fig.savefig(filename, bbox_inches='tight')

    if display:
        plt.show()
    else:
        plt.close()


def plot_series(t, series, title: Optional[str] = None, filename: Optional[str] = None, display: Optional[bool] = False):
    fig, ax = plt.subplots(figsize=(8, 5))

    data = series

    if type(series) is not list:
        if type(series) is tuple:
            data = [series]
        else:
            data = [(series, None)]

    for (d, l) in data:
        plt.plot(t, d, label=l, linewidth=2.6)
    plt.legend()
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x(t)$")
    plt.setp(ax.spines.values(), color='#374151')

    if filename:
        fig.savefig(filename, bbox_inches='tight')

    if display:
        plt.show()
    else:
        plt.close()


def plot_scatter(targets, predictions, filename: Optional[str] = None, display: Optional[bool] = False):
    fig = plt.figure(figsize=(8, 5))

    wTrue=targets[:,0]
    thetaTrue=targets[:,1]
    wPred=predictions[:,0]
    thetaPred=predictions[:,1]
    L = len(wTrue)
    
    ax1 = plt.subplot(1,2,1)
    ax1.scatter(range(L), wTrue, label='$\omega$ true')
    ax1.scatter(range(L), wPred, label='$\omega$ pred')
    ax1.legend()
    ax1.set_title('$\omega$ Predictions on Test Data', fontsize=16, pad=12)
    ax1.set_xlabel("trajectory number")
    ax1.set_ylabel("$\omega$")

    ax2 = plt.subplot(1,2,2)
    ax2.scatter(range(L), thetaTrue, label='$\\theta$ true')
    ax2.scatter(range(L), thetaPred, label='$\\theta$ pred')
    ax2.legend()
    ax2.set_title('$\\theta$ Predictions on Test Data', fontsize=16, pad=12)
    ax2.set_xlabel("trajectory number")
    ax2.set_ylabel("$\\theta$")

    if filename:
        fig.savefig(filename, bbox_inches='tight')

    if display:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    targets=np.random.normal(0,1,(20, 2))
    predictions=np.random.normal(0,1,(20, 2))
    plot_scatter(targets, predictions, display=True)
