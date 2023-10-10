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

def hard_routing(path_prices, max_price, demand):
    flows = np.zeros(len(path_prices))
    min_price = np.min(path_prices)
    if min_price <= max_price:
        best_path = np.argmin(path_prices)
        flows[best_path] = demand
    return flows

def soft_routing(path_prices, max_price, demand, sensitivity):
    flows = np.zeros(len(path_prices))
    path_prices_sorted = np.sort(path_prices)
    best_paths = np.argsort(path_prices)
    total_flow = 0
    for i in range(len(path_prices)):
        if path_prices_sorted[i] <= max_price:
            break
        remaining_flow = demand - total_flow
        if i+1 < len(path_prices):
            price_difference = (path_prices_sorted[i+1] - path_prices_sorted[i])/sensitivity
            if (i+1)*price_difference < remaining_flow:
                flows[:i+1] += price_difference
                total_flow += (i+1)*price_difference
        else:
            flows[:i+1] += remaining_flow/(i+1)
            break
    flows[best_paths] = flows.copy()
    return flows