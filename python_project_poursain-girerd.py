# upload the libraries to import data

import pandas as pd 
# this library is for data manipulation 
import numpy as np
# thi library is for data plotting and matrix manipulation 
import pandas_datareader.data as pdr
import yfinance as yf
# this library is ment for finiding the data of yahoo finance mainly refering to stocks
yf.pdr_override()
from datetime import datetime

def get(tickers, startdate, enddate):
    def data(ticker):
        return pdr.get_data_yahoo(ticker, start=startdate, end=enddate)
    datas = map (data, tickers)
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

# if you want to add more assets you just nead to add them in the list of tickets
tickers = ['AIR.PA','AI.PA','^FCHI']
#tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']
all_data = get(tickers, datetime(2018, 1, 1), datetime(2023, 1, 1))

# Frequency Distributions on closing prices only 
#------------------------------------------------------------------------------
# Import matplotlib to crate static, animated , and interactive visulaisation in phyton
import matplotlib.pyplot as plt

# Isolate the `Adj Close` values and transform the DataFrame
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')

# Calculate the daily percentage change for `daily_close_px`= RETURNS 
daily_pct_change = daily_close_px.pct_change()

# print the obtained returns 
print (daily_pct_change)
# Plot the distributions
daily_pct_change.hist(bins=25, sharex=False, figsize=(14,10))

# Show the resulting plot
plt.show()

# Moving average on closing prices only - AIRBUS
#----------------------------------------------------------------------------
#import pandas_datareader.data as pdr
#import yfinance as yf
#yf.pdr_override()
#from datetime import datetime
air = pdr.get_data_yahoo('AIR.PA',
                          start=datetime(2018, 1, 1),  ##(yyyy, dd, mm)
                          end=datetime(2023, 1, 1))     ##(yyyy, dd, mm)
print(air)

# Isolate the adjusted closing prices
adj_close_px = air[['Adj Close']]

# Calculate the moving average for aapl
moving_avg = adj_close_px.rolling(window=40).mean()

# Inspect the result
print(moving_avg[-10:])

#PLOT THE MOOVING AVERAGE

# Short moving window rolling mean
air['42'] = adj_close_px.rolling(window=40).mean()

# Long moving window rolling mean
air['252'] = adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
air[['Adj Close', '42', '252']].plot()

# Show plot
plt.show()

# Moving average on closing prices only - AIR LIQUIDE
#----------------------------------------------------------------------------
#import pandas_datareader.data as pdr
#import yfinance as yf
#yf.pdr_override()
#from datetime import datetime
ai = pdr.get_data_yahoo('AI.PA',
                          start=datetime(2018, 1, 1),  ##(yyyy, dd, mm)
                          end=datetime(2023, 1, 1))     ##(yyyy, dd, mm)
print(ai)

# Isolate the adjusted closing prices
adj_close_px = ai[['Adj Close']]

# Calculate the moving average for aapl
moving_avg = adj_close_px.rolling(window=40).mean()

# Inspect the result
print(moving_avg[-10:])

#PLOT THE MOOVING AVERAGE

# Short moving window rolling mean
ai['42'] = adj_close_px.rolling(window=40).mean()

# Long moving window rolling mean
ai['252'] = adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
ai[['Adj Close', '42', '252']].plot()

# Show plot
plt.show()

# Moving average on closing prices only - CAC 40
#----------------------------------------------------------------------------
#import pandas_datareader.data as pdr
#import yfinance as yf
#yf.pdr_override()
#from datetime import datetime
cac = pdr.get_data_yahoo('^FCHI',
                          start=datetime(2018, 1, 1),  ##(yyyy, dd, mm)
                          end=datetime(2023, 1, 1))     ##(yyyy, dd, mm)
print(cac)

# Isolate the adjusted closing prices
adj_close_px = cac[['Adj Close']]

# Calculate the moving average for aapl
moving_avg = adj_close_px.rolling(window=40).mean()

# Inspect the result
print(moving_avg[-10:])

#PLOT THE MOOVING AVERAGE

# Short moving window rolling mean
cac['42'] = adj_close_px.rolling(window=40).mean()

# Long moving window rolling mean
cac['252'] = adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
cac[['Adj Close', '42', '252']].plot()

# Show plot
plt.show()

# VOLATILITY CALCULATION
#----------------------------------------------------------------------------
#Import matplotlib
#import matplotlib.pyplot as plt

# Define the minumum of periods to consider
# we choose to take a smale value as for the min_period in order to get a 
# better graphical rappresentation of the volatilities
min_periods = 65

# Calculate the volatility
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
#vol = daily_pct_change['AAPL'].rolling(min_periods).std() * np.sqrt(min_periods)

# Plot the volatility
vol.plot(figsize=(10, 8))

# Show the plot
plt.show()

# OLS regression - between ARIBUS AND AIR LIQUIDE

# Import the `api` model of `statsmodels` under alias `sm`
#! pip install datetools
import statsmodels.api as sm

# Import the `datetools` module from `pandas`
# from pandas.core import datetools
#If this does not work:
#! pip install datetools
# Isolate the adjusted closing price
all_adj_close = all_data[['Adj Close']]

# Calculate the returns
all_returns = np.log(all_adj_close / all_adj_close.shift(1))

# Isolate the AIRBUS returns
air_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AIR.PA']
# to drop index "Ticker"
air_returns.index = air_returns.index.droplevel('Ticker')

# Isolate the AIR LIQUIDE returns
ai_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AI.PA']
ai_returns.index = ai_returns.index.droplevel('Ticker')

# Build up a new DataFrame with AIRBUS and AIR LIQUIDE returns
return_data = pd.concat([air_returns, ai_returns], axis=1)[1:]
#[1:] to not get row 1 // axis=1 means take columns

return_data.columns = ['AIR.PA', 'AI.PA']       # rename columns

# Add a constant
X = sm.add_constant(return_data['AIR.PA'])

# Construct the model
model = sm.OLS(return_data['AI.PA'],X).fit()

# Print the summary
print(model.summary())

# Plotting the OLS Regression - for AIRBUS and AIR LIQUIDE
#----------------------------------------------------------------------------
# Import matplotlib
import matplotlib.pyplot as plt

# Plot returns of CAC 40 and AIRBUS
plt.plot(return_data['AIR.PA'], return_data['AI.PA'], 'b.')

# Add an axis to the plot
ax = plt.axis()

# Initialize `x`
x = np.linspace(ax[0], ax[1] + 0.01)
# x will help me to plot OLS regression // here x varies between min and max+0.01

# Plot the regression line
plt.plot(x, model.params[0] + model.params[1] * x, 'r', lw=2)
# my OLS regression line : y = 0.0006 + 0.3791 * x

# Customize the plot
plt.grid(True)
plt.axis('tight')  # axis are just large enough to show all data
plt.xlabel('AIRBUS Returns')
plt.ylabel('AIR LIQUID returns')

# Show the plot
plt.show()

# it is interesting to compare in a linear regression two differents stocks to have a representation 
# of their correlation or relationship of their trend during a specific period of time. 
# For example the COVID 19 pandemic crisis.

#SMA Strategy: creating signals refered to AIRBUS
#-----------------------------------------------------------------------------

# The strategy involves comparing a short-term moving average of the stock price 
# to a longer-term moving average, and generating buy or sell signals based on 
# the relative position of the two moving averages.

# Initialize the short and long windows

# THE STRATEGY 
# we choosed short window of 20 to respond more quickly to stock price changes and 
# and a large long window to balance it, and to respond more slowly to the changes   
short_window = 20
long_window = 150

# Initialize the `signals` DataFrame with the `signal` column
signals = pd.DataFrame(index=air.index)
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] = air['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = air['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
> signals['long_mavg'][short_window:], 1.0, 0.0)   

# Generate trading orders
signals['positions'] = signals['signal'].diff()

# Print `signals`
print(signals)

# Plot AIRBUS signals 
#-----------------------------------------------------------------------------
# Import `pyplot` module as `plt`
import matplotlib.pyplot as plt

# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in €')

# Plot the closing price
air['Close'].plot(ax=ax1, color='c', lw=2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals (magenta)
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='y')

# Plot the sell signals (black)
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='g')

# Show the plot
plt.show()

#SMA Strategy: creating signals refered to AIR LIQUIDE
#-----------------------------------------------------------------------------
# Initialize the short and long windows

short_window = 20
long_window = 150

# Initialize the `signals` DataFrame with the `signal` column
signals = pd.DataFrame(index=air.index)
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] = ai['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = ai['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
> signals['long_mavg'][short_window:], 1.0, 0.0)  

# Generate trading orders
signals['positions'] = signals['signal'].diff()

# Print `signals`
print(signals)

# Plot AIR LIQUIDE signals 
#-----------------------------------------------------------------------------
# Import `pyplot` module as `plt`
import matplotlib.pyplot as plt

# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in €')

# Plot the closing price
air['Close'].plot(ax=ax1, color='c', lw=2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals (magenta)
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='y')

# Plot the sell signals (black)
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='g')

# Show the plot
plt.show()

 
