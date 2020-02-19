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
            if(isinstance(candidates[i], int)):
                candidate_matrix[j][i] = float(candidates[i])
            else:
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

        for index in range(len(big_indices_mat)):
            for index2 in range(len(big_indices_mat[index])):
                big_indices_mat[index][index2] = theta[index2][big_indices[index]]

        eta_big, r, rr, s = np.linalg.lstsq(np.transpose(big_indices_mat), x_dot, rcond=None)

        for index in range(len(eta_big)):
            eta[big_indices[index]] = eta_big[index]

        error = np.linalg.norm(eta - eta_temp)

    terms = np.count_nonzero(eta)
    return eta, terms

def sym_power(x, n):
    power = 1
    for i in range(1, n):
        power = power * x
    return power


def stls_optimal_threshold(theta, x_dot, thresholds):
    term_list = []
    for threshold in thresholds:
        _, terms = stls(theta, x_dot, threshold)
        term_list.append(terms)

    pyplot.plot(thresholds, term_list)
    pyplot.xlabel("Thresholds")
    pyplot.ylabel("Number of terms")
    pyplot.show()

    return 0


def print_equation(coefficients, functions):
    equation = 0
    for coeff_index in range(len(coefficients)):
        equation += coefficients[coeff_index] * functions[coeff_index]

    print(equation)

    return equation


def plot_estimation(equation, actual):
    price_estimation = []
    for time in range(len(actual)):
        price_estimation.append(float(equation.subs(x, time)))
    return price_estimation


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
    candidate_matrix = []

    for i in range(1, 50):
        candidate_matrix.append(sym_power(x, i))

    candidate_matrix_values = construct_candidate_matrix(candidate_matrix, aapl_log_returns)

    thresholds = []
    for n in np.linspace(-5, 1, 10):
        thresholds.append(np.float_power(10, n))

    stls_optimal_threshold(candidate_matrix_values, aapl_log_returns_dot, thresholds)

    # eta, _ = stls(candidate_matrix_values, aapl_log_returns_dot, 2.18)
    #
    # equation = print_equation(eta, candidate_matrix)
    # price_estimation = plot_estimation(equation, aapl_log_returns)
    #
    #
    # pyplot.plot(aapl_log_returns)
    # pyplot.plot(price_estimation)
    # pyplot.xlabel("Time unit")
    # pyplot.ylabel("Prices")
    # pyplot.show()
    #
