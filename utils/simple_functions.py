import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def total_average(x):
    averaged_vec = np.zeros_like(x)
    averaged_vec[0] = x[0]
    partial_sum = 0
    for i in range(len(x)):
        partial_sum += x[i]
        averaged_vec[i] = partial_sum/(i + 1.0)
    return averaged_vec

def softmax(x, c):
    if type(x) == list:
        y = np.array(x)
    else:
        y = x.copy()
    y = np.exp((y - np.max(y))/c)
    f_x = y / np.sum(y)
    return f_x