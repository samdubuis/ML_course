'''
Functions used throughout the project are repertoried here
'''
import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def expansion(x,degree=1):
    #polynomial expansion
    X_ex=x
    for deg in range(2, degree):
        X_ex = np.c_[X_ex, np.power(x, deg)]
    #cross term
    for i in range(0, x.shape[1]):
        for j in range(i+1, x.shape[1]):
            X_ex = np.c_[X_ex, np.multiply(x[:,i],x[:,j])]
    return X_ex