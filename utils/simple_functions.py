import numpy as np
import logging
import sys

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
    path_prices_sorted = np.append(path_prices_sorted, [np.inf])
    best_paths = np.argsort(path_prices)
    total_flow = 0
    logging.debug("The path prices are %s and the maximum price is %f", path_prices, max_price)
    if path_prices_sorted[0] > max_price:
        logging.debug("path price too high. Path price: %f, Max price:, %f", path_prices_sorted[0], max_price)
        pass
    else:
        for i in range(len(path_prices)):
            remaining_flow = demand - total_flow
            logging.debug("In iteration %d of soft routing with %f flow remaining", i, remaining_flow)
            price_difference = (min(path_prices_sorted[i+1], max_price) - path_prices_sorted[i])/sensitivity
            if  remaining_flow < (i+1)*price_difference:
                flows[:i+1] += remaining_flow/(i+1)
                logging.debug("filled upto maximum demand, breaking the loop")
                break
            else:
                flows[:i+1] += price_difference
                total_flow += (i+1)*price_difference
                if path_prices_sorted[i+1] > max_price:
                    logging.debug("reached price ceiling, breaking the loop")
                    break
                else:
                    logging.debug("filled upto price difference, moving to next iteration")

        flows[best_paths] = flows.copy()
    return flows

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    path_prices = [0.2, 0.1]
    max_price = 0.5
    demand = 10.0
    sensitivity = 0.1
    flows = soft_routing(path_prices, max_price, demand, sensitivity)
    print(flows)