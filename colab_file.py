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
from ta import add_all_ta_features
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from ta.utils import dropna
import yfinance as yf
import datetime
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA


def cumu_sum():
    test_end_date = dt.now()
    test_start_date = dt(test_end_date.year, test_end_date.month - 3, test_end_date.day)
    train_end_date = dt(test_end_date.year, test_end_date.month - 3, test_end_date.day)
    train_start_date = dt(test_end_date.year - 2, test_end_date.month, test_end_date.day)
    print("Train Data dates : ", train_start_date, 'to', train_end_date)
    print("Test Data dates : ", test_start_date, 'to', test_end_date)

    start = datetime.datetime(2012, 4, 6)
    end = datetime.datetime(2022, 4, 6)

    stock = yf.download(VOL[0], start, end)
    stock = dropna(stock)
    df = add_all_ta_features(
        stock, open="Open", high="High", low="Low", close="Close", volume="Volume")

    df = df.drop(["trend_psar_down", "trend_psar_up"], axis=1)
    df = df.dropna()

    y = df['Close']
    X = df.drop(columns=['Close'])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=0)
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)

    X_test = sc.transform(X_test)

    components = 1
    explained_variance = 0
    while np.sum(explained_variance) < .95:
        pca = PCA(n_components=components)
        X_train_fit = pca.fit_transform(X_train)
        X_test_fit = pca.transform(X_test)

        explained_variance = pca.explained_variance_ratio_
        components += 1

    refit_data = pd.DataFrame(data=X_train_fit, columns=[f'PC{num + 1}' for num, p, in enumerate(explained_variance)])

    refit_data['Label'] = np.array(y_train)

    sum_eigenvalues = np.cumsum(explained_variance)

    plt.bar(range(0, len(explained_variance)), explained_variance,
            alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0, len(sum_eigenvalues)), sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.title('Component Count vs Explained Variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    print(f'Optimal Principal Component Value: {len(explained_variance)}')


def pair():
    from scipy.sparse import data

    test_end_date = dt.now()
    test_start_date = dt(test_end_date.year, test_end_date.month - 3, test_end_date.day)
    train_end_date = dt(test_end_date.year, test_end_date.month - 3, test_end_date.day)
    train_start_date = dt(test_end_date.year - 2, test_end_date.month, test_end_date.day)
    print("Train Data dates : ", train_start_date, 'to', train_end_date)
    print("Test Data dates : ", test_start_date, 'to', test_end_date)

    start = datetime.datetime(2012, 4, 6)
    end = datetime.datetime(2022, 4, 6)

    stock = yf.download(VOL[0], start, end)
    stock = dropna(stock)
    df = add_all_ta_features(
        stock, open="Open", high="High", low="Low", close="Close", volume="Volume")

    df = df.drop(["trend_psar_down", "trend_psar_up"], axis=1)
    df = df.dropna()

    y = df['Close']
    X = df.drop(columns=['Close'])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=0)
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)

    X_test = sc.transform(X_test)

    components = 3
    explained_variance = 0

    pca = PCA(n_components=components)
    X_train_fit = pca.fit_transform(X_train)
    X_test_fit = pca.transform(X_test)

    explained_variance = pca.explained_variance_ratio_

    refit_data = pd.DataFrame(data=X_train_fit, columns=[f'PC{num + 1}' for num, p, in enumerate(explained_variance)])

    refit_data['Label'] = np.array(y_train)

    sum_eigenvalues = np.cumsum(explained_variance)

    sns.pairplot(refit_data, vars=refit_data.columns[:-1],
                 hue='Label', diag_kws=dict(color=".2", hue=None))
    plt.show()




    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    from scipy.sparse import data

    test_end_date = dt.now()
    test_start_date = dt(test_end_date.year, test_end_date.month - 3, test_end_date.day)
    train_end_date = dt(test_end_date.year, test_end_date.month - 3, test_end_date.day)
    train_start_date = dt(test_end_date.year - 2, test_end_date.month, test_end_date.day)
    print("Train Data dates : ", train_start_date, 'to', train_end_date)
    print("Test Data dates : ", test_start_date, 'to', test_end_date)

    start = datetime.datetime(2012, 4, 6)
    end = datetime.datetime(2022, 4, 6)

    stock = yf.download('AAPL', start, end)
    stock = dropna(stock)
    df = add_all_ta_features(
        stock, open="Open", high="High", low="Low", close="Close", volume="Volume")

    df = df.drop(["trend_psar_down", "trend_psar_up"], axis=1)
    df = df.dropna()

    sc = StandardScaler()

    y = df['Close']

    X = df.drop(columns=['Close'])

    X = sc.fit_transform(X)
    y = sc.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=0)

    # X_train = sc.fit_transform(X_train)

    # X_test = sc.transform(X_test)

    components = 23

    pca = PCA(n_components=components)
    X_train_fit = pca.fit_transform(X_train)
    X_test_fit = pca.transform(X_test)
    X_train_fit = X_train_fit.reshape((X_train_fit.shape[0], X_train_fit.shape[1], 1))
    explained_variance = pca.explained_variance_ratio_

    # refit_data = pd.DataFrame(data=X_train_fit, columns=[f'PC{num+1}' for num, p, in enumerate(explained_variance)])

    # refit_data['Closing Price'] = np.array(y_train)

    model = Sequential()

    model.add(LSTM(units=120, return_sequences=True))
    model.add(LSTM(units=60, return_sequences=True))
    model.add(LSTM(units=30))
    model.add(Dense(units=1, activation="linear"))

    model.compile(loss="mean_squared_error", optimizer="RMSprop", metrics=["accuracy"])

    model.fit(X_train_fit, y_train, epochs=10)

    X_test_fit = X_test_fit.reshape(X_test_fit.shape[0], X_test_fit.shape[1], 1)
    pred_values = model.predict(X_test_fit)
    # Y_test = sc.transform(np.array(y_test).reshape(-1,1))
    # pred_values = sc.inverse_transform(pred_values)

    y_test = np.array(y_test)
    plt.plot(pred_values)
    plt.plot(y_test)
