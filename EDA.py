import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_datareader import DataReader
from datetime import datetime as dt
SHOW = True
VOL = ['WEI', 'IVDA', 'TKAT', 'CYRN', 'SLS',
       'XPON', 'HUSN', 'SNMP', 'GREE', 'USX']


def high_low_dist(data):

    stock, df = data
    dist = df['High'] - df['Low']
    plt.plot(dist)
    plt.xlabel('Dates')
    plt.ylabel('Distance in $ between Stock High and Low')
    plt.title(f'Volatility in {stock} in a given day')

    if SHOW:
        plt.show()


def open_close_dist(data):

    stock, df = data
    dist = abs(df['Open'] - df['Close'])
    plt.plot(dist)
    plt.xlabel('Dates')
    plt.ylabel('Distance in $ between Stock Open and Close')
    plt.title(f'Volatility in {stock} in a given day')

    if SHOW:
        plt.show()


def volatility_vs_volume(data):

    stock, df = data

    open_close = abs(df['Open'] - df['Close'])
    high_low = df['High'] - df['Low']

    avg_volatility = (open_close + high_low) / 2

    plt.scatter(avg_volatility, df['Volume'])
    plt.xlabel('Average Volatility')
    plt.ylabel('Stock Volume Traded')
    plt.title(f'Volatility in {stock} vs Volume')

    if SHOW:
        plt.show()


def pair_plot(data, chosen=None):

    stock, df = data

    selected = ['High', 'Low'] if chosen is None else chosen
    sns.pairplot(data=df, vars=selected)

    if SHOW:
        plt.show()


def heatmap(data, chosen=None):
    ''' '''

    stock, df = data
    selected = ['High', 'Low'] if chosen is None else chosen

    sns.heatmap(df[selected].corr(), annot=True, cmap='Blues_r')
    plt.title('Correlation of '+' vs '.join([stat for
                                             stat in selected])+f' in {stock}')
    if SHOW:
        plt.show()


def box_plot(data, chosen=None):
    ''' '''

    selected = 'High' if chosen is None else chosen
    df_lst = [df[selected] for stock, df in data]
    stk_lst = [stock for stock, df in data]

    df = pd.DataFrame()
    for data, stock in zip(df_lst, stk_lst):
        df[stock] = data
    df.dropna(inplace=True)
    sns.boxplot(data=df)

    plt.xlabel('Stocks')
    plt.ylabel(f'Value for Statistic: {selected}')

    plt.title(f'Boxplot of Stocks Compared Using Statistic: {selected} ')

    if SHOW:
        plt.show()


def main():

    test_end_date = dt.now()
    test_start_date = dt(test_end_date.year, test_end_date.month - 3, test_end_date.day)
    train_end_date = dt(test_end_date.year, test_end_date.month - 3, test_end_date.day)
    train_start_date = dt(test_end_date.year - 2, test_end_date.month, test_end_date.day)
    print("Train Data dates : ", train_start_date, 'to', train_end_date)
    print("Test Data dates : ", test_start_date, 'to', test_end_date)

    data_frames = {}
    for stock in VOL:
        data_frames[stock] = DataReader(stock, 'yahoo', test_start_date, test_end_date)

    example = list(data_frames.items())[0]

    high_low_dist(example)
    open_close_dist(example)

    volatility_vs_volume(example)

    pair_plot(example, ['Open', 'Close'])
    heatmap(example, ['High', 'Low', 'Volume'])

    box_plot(list(data_frames.items()))



main()


