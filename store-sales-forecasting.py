import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

rawData = pd.read_csv('train-store-data.csv')
print(rawData.head())
print(rawData.describe())

# Time Alterations
rawData['Order Date'] = pd.to_datetime(rawData['Order Date'], format='%d/%m/%Y')
rawData['Ship Date'] = pd.to_datetime(rawData['Ship Date'], format='%d/%m/%Y')
print(rawData.info())
print(rawData['Order Date'].describe())

rawData.drop('Row ID',axis = 1, inplace = True)
rawData.sort_values(by=['Order Date'], inplace=True, ascending=True)
rawData.set_index("Order Date", inplace = True)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
print(rawData.head())

# Forecast features
sales_forcasting_df = rawData['Sales']
print(sales_forcasting_df.head())

# Prediction features
predFeatures = ['Ship Mode', 'Order Date', 'Segment', 'Country', 'State', 'Region', 'Sub-Category', 'Category', 'Sales']

predDF = rawData[predFeatures].copy(deep = True)
predDF['logSales'] = np.log(predDF['Sales'])

# Plotting setup
colors = ["#2B4162", "#D7B377", "#606C38", "#ADB5BD", "#385F71", "#8F754F", "#283618", "#495057"]
sns.set_theme(style="whitegrid", palette = sns.color_palette(colors))

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

sales_forcasting_df.plot()
plt.show()
