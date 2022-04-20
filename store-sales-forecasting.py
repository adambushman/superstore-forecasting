import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

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
decompose_result.plot()
plt.show()


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

# Forecast features
forFeatures = ['Order Date']
forResponse = ['Sales']

# Sales Forecast plot
sns.boxplot(x = 'Category', y = 'Sales', data = predDF).set_title(label = "Monthly Sales | 2015-2019")
plt.show()


