import numpy as np
import scipy.io

def split_data(fname='face'):

    """
    Function for splitting data.
    Parameters
    ----------
    fname: str
        Name of the '.mat' input file
    -------
    data: dict
        * train: tuple
            - X: features
            - y: labels
        * test: tuple
            - X: features
            - y: labels
    """

    # load '.mat' file
    data = scipy.io.loadmat(fname)

    # Images
    # N: number of images
    # D: number of pixels
    X = data['X']  # shape: [D x N]
    y = data['l']  # shape: [1 x N]
    assert(X.shape[1] == y.shape[1])

    D, N = X.shape

    # Partition dataset to train and test sets
    # Img set
    X_train, X_test = X[:, 0:320], X[:, 320:520]
    # Label set
    y_train, y_test = y[:, 0:320], y[:, 320:520]

    return {'train': (X_train, y_train), 'test': (X_test, y_test)}

