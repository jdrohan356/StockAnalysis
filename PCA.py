from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_datareader import DataReader
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
# from ta import add_all_ta_features
# import yfinance as yf

SHOW = False
VOL = ['DOGZ', 'IVDA', 'TKAT', 'CYRN', 'SLS',
       'XPON', 'HUSN', 'SNMP', 'GREE', 'USX']



# def organize_groups(x_groups, y_groups, i):
#     ''' Organizes the groups for K-fold cross validation '''
#
#     x_train = [tup[1] for tup in tuple(enumerate(x_groups)) if tup[0] != i]
#     x_train = np.concatenate(x_train)
#
#     x_test = [tup[1] for tup in tuple(enumerate(x_groups)) if tup[0] == i]
#     x_test = np.concatenate(x_test)
#
#     y_train = [tup[1] for tup in tuple(enumerate(y_groups)) if tup[0] != i]
#     y_train = np.concatenate(y_train)
#
#     y_test = [tup[1] for tup in tuple(enumerate(y_groups)) if tup[0] == i]
#     y_test = np.concatenate(y_test)
#
#     # Creates two dataframes of information based on the organized groups
#     train = pd.DataFrame()
#     train['x'] = x_train
#     train['y'] = y_train
#
#     test = pd.DataFrame()
#     test['x'] = x_test
#     test['y'] = y_test
#
#     return train, test



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
        break

    name, df = list(data_frames.items())[0][0], list(data_frames.items())[0][1]

    # stock = yf.download(name, train_start_date, train_end_date)
    # stock = dropna(stock)
    # df = add_all_ta_features(
    #     stock, open="Open", high="High", low="Low", close="Close", volume="Volume")
    #
    # df = df.drop(["trend_psar_down", "trend_psar_up"], axis=1)
    # df = df.dropna()

    # WFTUF

    # y = df['Close']
    # plt.plot(y)
    # plt.title(f'Closing Price of {name} Over Time')
    # plt.xlabel('Dates')
    # plt.ylabel('Stock Price')
    # plt.show()
    # X = df.drop(columns=['Close'])
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                    test_size=0.2, random_state=0)
    # sc = StandardScaler()
    #
    # X_train = sc.fit_transform(X_train)
    #
    # X_test = sc.transform(X_test)
    #
    # components = 1
    #
    # # while pca.explained_variance_ratio_ < .95:
    #
    #
    # pca = PCA(n_components=components)
    # X_train_fit = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    # print(X_train_fit)
    # # y = StandardScaler()
    # # x = y.fit(X_train_fit)
    # # print(y.inverse_transform(X_train_fit))
    #
    #
    #
    vals = np.array(pca.components_).T
    print(vals)
    print(pd.DataFrame(vals, columns=['PC-1', 'PC-2'], index=X.columns))
    if SHOW:
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
        plt.show()
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        plt.show()
    #
    # explained_variance = pca.explained_variance_ratio_
    #
    # # print(df.columns)
    # # # Calculate the variance explained by priciple components
    # # print('Variance of each component:', pca.explained_variance_ratio_)
    # # print('\n Total Variance Explained:',
    # #       round(sum(list(pca.explained_variance_ratio_)) * 100, 2), '\n')
    #
    #
    # # Create a new dataset from principal components
    # refit_data = pd.DataFrame(data=X_train,
    #                   columns=[f'PC{num+1}' for num, p, in enumerate(explained_variance)])
    #
    #
    #
    # refit_data['Label'] = np.array(y_train)
    # print(refit_data.head(5))
    #
    # #
    # # sum_eigenvalues = np.cumsum(explained_variance)
    # # if SHOW:
    # #     plt.bar(range(0, len(explained_variance)), explained_variance,
    # #             alpha=0.5, align='center', label='Individual explained variance')
    # #     plt.step(range(0, len(sum_eigenvalues)), sum_eigenvalues, where='mid',
    # #              label='Cumulative explained variance')
    # #     plt.ylabel('Explained variance ratio')
    # #     plt.xlabel('Principal component index')
    # #     plt.legend(loc='best')
    # #     plt.tight_layout()
    # #     plt.show()
    #
    #
    # # bin_values(y_train)
    # # model = ExtraTreesClassifier(n_estimators=3)
    # # model.fit(X_train, y_train)
    # # print(model.feature_importances_)
    #

main()




