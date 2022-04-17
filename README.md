# Financial Market Analysis

## Abstract
Financial Markets represent a type of time-series data that exhibits both seasonality and trend, requiring cumulation of multiple statistical approaches for machine learning modeling. The goal of our project is to analyze financial time-series datasets obtained from independent market sources such as Yahoo! Finance to build prediction models using supervised learning techniques.

One of the objectives of our project is to study the role of data features and the relative importance of features compared to one another. Financial time-series data can contain over 90 domain features created by market experts and financial analysts. Including several of these features into our stock dataframe, we want to check which features impact the prediction results of the supervised learning models the most.

We aim to build a LASSO regularization prediction model from scratch and tune it to our stock datasets. Once the model is created, we will examine the co-efficients assigned to each feature, and remove features with co-efficient of zero. The finalized list of features will be then fed to an LSTM- neural network to compare prediction performance. 

