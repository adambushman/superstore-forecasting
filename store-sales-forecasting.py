import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import math
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
print(rawData['Order Date'].describe())
print(rawData.info())
print(rawData.head())

# Log of Sales
rawData['logSales'] = np.log(rawData['Sales'])

# Forecast
group_by_date_sales = rawData[['orderYrMon', 'Sales']].groupby(['orderYrMon']).sum().reset_index()
group_by_date_sales.set_index('orderYrMon', inplace = True)
group_by_date_sales.index=pd.to_datetime(group_by_date_sales.index, format = '%Y-%m')

group_by_date_sales.index.freq = 'MS'
decompose_result = seasonal_decompose(group_by_date_sales, model='multiplicative')

train, test = group_by_date_sales.iloc[:42, 0], group_by_date_sales.iloc[42:, 0]  #train set 42/48 = 87.5%
model = ExponentialSmoothing(train, seasonal='mul',seasonal_periods=12, initialization_method='estimated').fit()
pred = model.predict(start=test.index[0], end=test.index[-1])
#predict (6 Months)

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
plt.plot(pred.index, pred, label='Holt-Winters', linewidth = 3)
plt.title("6 Month Sales Forecasting", fontsize = 20)
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()

#Forcasting Accuracy
print(f'Mean Absolute Error = {mean_absolute_error(test,pred)}')
print(f'Mean Squared Error = {mean_squared_error(test,pred)}')
print(f'Root Mean Squared Error ={math.sqrt(mean_squared_error(test,pred))}')
print(f'R^2 Score = {r2_score(test,pred)}\n')


# Prediction
predFeatures = ['Ship Mode', 'Order Date', 'Segment', 'Country', 'State', 'Region', 'Sub-Category', 'Category', 'logSales']
predDF = rawData[predFeatures].copy(deep = True)


# Sales vs Shipping Mode
sns.boxplot(x = 'Ship Mode', y = 'logSales', data = predDF).set_title(label = "Shipping Mode vs Sales [Log Scale]")
plt.show()

# Sales vs Segment
sns.boxplot(x = 'Segment', y = 'logSales', data = predDF).set_title(label = "Customer Segment vs Sales [Log Scale]")
plt.show()

# Sales vs Category
sns.boxplot(x = 'Category', y = 'logSales', data = predDF).set_title(label = "Product Category vs Sales [Log Scale]")
plt.show()

# Sales vs Region
sns.boxplot(x = 'Region', y = 'logSales', data = predDF).set_title(label = "Product Region vs Sales [Log Scale]")
plt.show()


# RFE
y = predDF['logSales']
xDum = pd.concat([pd.get_dummies(predDF['Category']), pd.get_dummies(predDF['Ship Mode']), pd.get_dummies(predDF['Segment']), pd.get_dummies(predDF['Region'])], axis = 1)
xLin = xDum.drop(columns=['Technology', 'Central', 'Consumer', 'First Class'])
xScale =  pd.DataFrame(data = StandardScaler().fit_transform(xDum), columns = xDum.columns)

# Linear model
linModel = LinearRegression()
rfeLin = RFECV(linModel)
rfeLinModel = rfeLin.fit(xLin, y)
scoreLin = rfeLinModel.score(xLin, y)
indxLin = [i for i, l in enumerate(list(rfeLinModel.support_)) if l == True]


# Decision tree model
dtModel = DecisionTreeRegressor()
rfeDT = RFECV(dtModel)
rfeDTModel = rfeDT.fit(xDum, y)
scoreDT = rfeDTModel.score(xDum, y)
indxDT = [i for i, l in enumerate(list(rfeDTModel.support_)) if l == True]


print('Score:', round(scoreLin,3),'| The top Linear features are:', list(xLin.columns[indxLin]))
print('Score:', round(scoreDT,3),'| The top Decision Tree features are:', list(xDum.columns[indxDT]))