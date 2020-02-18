import calendar
import datetime
import matplotlib.pyplot as pyplot
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


# Function that calls ARIMA model to fit and forecast the data
def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction


if __name__ == "__main__":

    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2019, 12, 31)
    # Obtain and plot the logarithmic returns of Amazon prices
    aapl_df = getStock("AAPL",start_date, end_date)
    aapl_log_returns = np.array(aapl_df["log_returns"])

    NumberOfElements = len(aapl_log_returns)

    # Use 70% of data as training, rest 30% to Test model
    TrainingSize = int(NumberOfElements * 0.7)
    TrainingData = aapl_log_returns[0:TrainingSize]
    TestData = aapl_log_returns[TrainingSize:NumberOfElements]


    min_error = 10
    p_best = 0
    q_best = 0
    for p in range(5):
        for q in range(5):
            if(p==0 and q==0):
                continue

            # new arrays to store actual and predictions
            Actual = [x for x in TrainingData]
            Predictions = list()

            # in a for loop, predict values using ARIMA model
            for t in range(len(TestData)):
                ActualValue = TestData[t]
                # forcast value
                Prediction = StartARIMAForecasting(Actual, p, 0, q)
                #print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
                # add it in the list
                Predictions.append(Prediction)
                Actual.append(ActualValue)

            # Print MSE to see how good the model is
            Error = mean_squared_error(TestData, Predictions)
            print("p = " + str(p))
            print("q = " + str(q))
            print('Test Mean Squared Error (smaller the better fit): %.6f' % Error)
            if(Error < min_error):
                min_error = Error
                p_best = p
                q_best = q

    print("p_best = " + str(p_best))
    print("q_best = " + str(q_best))

    # new arrays to store actual and predictions
    Actual = [x for x in TrainingData]
    Predictions = list()

    # in a for loop, predict values using ARIMA model
    for t in range(len(TestData)):
        ActualValue = TestData[t]
        # forcast value
        Prediction = StartARIMAForecasting(Actual, p_best, 0, q_best)
        # print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
        # add it in the list
        Predictions.append(Prediction)
        Actual.append(ActualValue)
    # plot
    pyplot.plot(TestData)
    pyplot.plot(Predictions, color='red')
    pyplot.show()


