import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import pandas_datareader as pdr
import datetime
import sympy as sym


def getStock(stock_name, start_date, end_date):
    """
    Download, calculate and plot the stocks logarithmic returns.
    """
    print("Downloading and plotting " + stock_name + " log returns...")
    stock = pdr.get_data_yahoo(stock_name, start_date, end_date)
    stock["log_returns"] = np.log(stock["Adj Close"]) - np.log(stock["Adj Close"].shift(1))
    stock.dropna(inplace=True)
    return stock


def construct_candidate_matrix(candidates, state):

    candidate_matrix = np.zeros((len(state), len(candidates)))
    for i in range(len(candidates)):
        for j in range(len(state)):
            candidate_matrix[j][i] = float(candidates[i].subs(x, state[j]))
    return candidate_matrix


def stls(theta, x_dot, threshold):

    eta, residuals, rank, singular = np.linalg.lstsq(theta, x_dot, rcond=None)

    error = 1
    while error > 0.00001:
        error_temp = error
        eta_temp = eta
        small_indices = [i for i, x in enumerate(eta) if x < threshold]
        big_indices = [i for i, x in enumerate(eta) if x >= threshold]

        for index_val in small_indices:
            eta[index_val] = 0

        big_indices_mat = np.zeros([len(big_indices), len(theta)])
        for index in range(len(big_indices)):
            big_indices_mat[index] = theta[big_indices[index]]

        print(big_indices_mat)

        eta_big, r, rr, s = np.linalg.lstsq(big_indices_mat, x_dot, rcond=None)



        for index in range(len(eta_big)):
            eta[big_indices[index]] = eta_big[index]

        print(eta - eta_temp)
        #error = np.linalg.norm(eta - eta_temp)

    return eta













if __name__ == "__main__":

    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2019, 12, 31)
    # Obtain and plot the logarithmic returns of Amazon prices
    aapl_df = getStock("AAPL", start_date, end_date)
    aapl_log_returns = np.array(aapl_df["log_returns"])

    aapl_log_returns_dot = np.gradient(aapl_log_returns)

    # pyplot.plot(aapl_log_returns)
    # pyplot.plot(aapl_log_returns_dot)
    # pyplot.show()

    x = sym.Symbol('x')

    candidate_matrix = [x, x*x, x*x*x, x*x*x*x]

    candidate_matrix_values = construct_candidate_matrix(candidate_matrix, aapl_log_returns)

    print(stls(candidate_matrix_values, aapl_log_returns_dot, 0.001))