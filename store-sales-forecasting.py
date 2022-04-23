import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import math
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sma

rawData = pd.read_csv('train-store-data.csv')

#print(rawData.head())
#print(rawData.describe())

#pd.set_option('display.expand_frame_repr', False)
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 1)

# Plotting setup
colors = ["#2B4162", "#D7B377", "#606C38", "#ADB5BD", "#385F71", "#8F754F", "#283618", "#495057"]
sns.set_theme(style="whitegrid", palette = sns.color_palette(colors))

# Time Alterations
rawData['orderYrMon'] = pd.to_datetime(rawData['Order Date']).dt.to_period('M').astype(str)
rawData['orderYear'] = pd.to_datetime(rawData['Order Date']).dt.year.astype(int)
rawData['orderQuarter'] = 'Q' + pd.to_datetime(rawData['Order Date']).dt.quarter.astype(str)
rawData['orderMonth'] = 'M' + pd.to_datetime(rawData['Order Date']).dt.month.astype(str)
rawData['logSales'] = np.log(rawData['Sales'])

print(rawData['Order Date'].describe())
print(rawData.info())
print(rawData.head())

# Forecast
group_by_date_sales = rawData[['orderYrMon', 'Sales']].groupby(['orderYrMon']).sum().reset_index()
group_by_date_sales.set_index('orderYrMon', inplace = True)
group_by_date_sales.index=pd.to_datetime(group_by_date_sales.index, format = '%Y-%m')

group_by_date_sales.index.freq = 'MS'
decompose_result = seasonal_decompose(group_by_date_sales, model='multiplicative')

m = 12
alpha = 1 / (2 * m)

train, test = group_by_date_sales.iloc[:42, 0], group_by_date_sales.iloc[42:, 0]  #train set 42/48 = 87.5%
modelS = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False, use_brute=True)
modelE = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12, initialization_method='estimated').fit()

predS = modelS.predict(start=test.index[0], end=test.index[-1])
predE = modelE.predict(start=test.index[0], end=test.index[-1])

# Time-series plot
ax = sns.lineplot(data = group_by_date_sales, x = 'orderYrMon', y = 'Sales', linewidth = 3)
ax.set_title(label = "4 Year Sales by Month", fontsize = 20)
ax.set(xlabel=None)
plt.xticks(rotation=45)
plt.show()

# Decomposition plot
decompose_result.plot()
plt.show()

#Forcasting plot
plt.plot(train.index, train, label='Train', linewidth = 3)
plt.plot(test.index, test, label='Test', linewidth = 3)
plt.plot(predE.index, predE, label='Holt-Winters', linewidth = 3)
plt.plot(predS.index, predS, label='Simple Exp', linewidth = 3)
plt.title("6 Month Sales Forecasting", fontsize = 20)
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()

#Forcasting Accuracy
#print(f'R^2 [Holt-Winters] = {r2_score(test,predE)}')
#print(f'R^2 [Simple Exp] = {r2_score(test,predS)}')
print(f'Mean Squared Error [Holt-Winters] = {mean_squared_error(test,predE)}')
print(f'Mean Squared Error [Simple Exp] = {mean_squared_error(test,predS)}')



# Prediction
predFeatures = ['Segment', 'Ship Mode', 'Region', 'Category', 'orderYear', 'orderMonth', 'orderQuarter', 'Sales']
predMinus = ['Segment', 'Ship Mode', 'Region', 'Category', 'orderYear', 'orderMonth', 'orderQuarter']
predDF = rawData[predFeatures].groupby(predMinus).sum('Sales').reset_index()
predDF['logSales'] = np.log(predDF['Sales'])


# Sales vs Shipping Mode
sns.boxplot(x = 'Ship Mode', y = 'logSales', data = predDF).set_title(label = "Shipping Mode vs Monthly Sales [Log Scale]")
plt.show()

# Sales vs Segment
sns.boxplot(x = 'Segment', y = 'logSales', data = predDF).set_title(label = "Customer Segment vs Monthly Sales [Log Scale]")
plt.show()

# Sales vs Category
sns.boxplot(x = 'Category', y = 'logSales', data = predDF).set_title(label = "Product Category vs Monthly Sales [Log Scale]")
plt.show()

# Sales vs Region
sns.boxplot(x = 'Region', y = 'logSales', data = predDF).set_title(label = "Product Region vs Monthly Sales [Log Scale]")
plt.show()


# RFE
y = predDF['logSales']
xDum = dataDums = pd.concat([predDF['orderYear'], pd.get_dummies(predDF['Category']), pd.get_dummies(predDF['Ship Mode']), pd.get_dummies(predDF['Segment']), pd.get_dummies(predDF['Region']), pd.get_dummies(predDF['orderQuarter']), pd.get_dummies(predDF['orderMonth'])], axis = 1)
xLin = xDum.drop(columns=['Technology', 'Central', 'Consumer', 'First Class', 'Q1', 'M1'])

# Linear model
linModel = LinearRegression()
rfeLin = RFECV(linModel)
rfeLinModel = rfeLin.fit(xLin, y)

# Decision tree model
dtModel = DecisionTreeRegressor()
rfeDT = RFECV(dtModel)
rfeDTModel = rfeDT.fit(xDum, y)

print('The top Linear features are, in order:')
linFeat = []
for i in rfeLinModel.ranking_:
    linFeat.append(xLin.columns[i])
print(linFeat)
print('The top Linear features are, in order:', rfeDTModel.ranking_)
dtFeat = []
for i in rfeDTModel.ranking_:
    dtFeat.append(xDum.columns[i])
print(dtFeat)


# Training, testing sets
x = xLin[['Q3', 'Q4', 'West', 'East', 'Home Office', 'Corporate', 'Standard Class', 'Second Class', 'Office Supplies', 'orderYear']]
x['const'] = x.apply(lambda x: 1, axis= 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, shuffle=False)


# Linear Regression
linModel = sma.OLS(y_train, x_train).fit()
print(linModel.summary())
yhat_lin = linModel.predict(x_test)
print('R^2 of actual vs test:', r2_score(y_test, yhat_lin))