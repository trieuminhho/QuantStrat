import calendar
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def getStock(stock_name,start_date, end_date):
    """
    Download, calculate and plot the stocks logarithmic returns.
    """
    print("Downloading and plotting " + stock_name + " log returns...")
    stock = pdr.get_data_yahoo(stock_name, start_date, end_date)
    stock["log_returns"] = np.log(stock["Adj Close"]) - np.log(stock["Adj Close"].shift(1))
    stock.dropna(inplace=True)
    return stock



if __name__ == "__main__":

    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2019, 12, 31)
    # Obtain and plot the logarithmic returns of Amazon prices
    aapl_df = getStock("AAPL",start_date, end_date)
    aapl_log_returns = np.array(aapl_df["log_returns"])

    # # forecast window length
    # windowLength = 20
    # forecastLength = aapl_log_returns.size - windowLength
    #
    # for day in range(forecastLength):
    #
    #     aapl_log_returns_offset = aapl_log_returns[(1+day):(windowLength+day)]
    #
    #     #fitting arima model
    #     for p in range(5):
    #         for q in range(5):
    #
    #             if(p==0 and q==0):
    #                 continue
    #             # fit model
    #             model = ARIMA(aapl_log_returns_offset, order=(p,0,q))
    #
    #             model_fit = model.fit(disp=0)
    #             print(model_fit.resid)
    #
    #             # plot residual errors
    #             residuals = DataFrame(model_fit.resid)
    #             residuals.plot()
    #             pyplot.show()
    #             residuals.plot(kind='kde')
    #             pyplot.show()
    #             print(residuals.describe())


    size = int(len(aapl_log_returns) * 0.9)
    train, test = aapl_log_returns[0:size], aapl_log_returns[size:len(aapl_log_returns)]
    history = [aapl_log_returns for aapl_log_returns in train]
    predictions = list()

    print("size " + str(size))
    print("train size" + str(len(train)))
    print("test size " + str(len(test)))
    print(history)




    for t in range(len(test)):
        model = ARIMA(history, order=(5, 0, 1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()