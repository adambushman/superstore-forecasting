import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

rawData = pd.read_csv('train-store-data.csv')
print(rawData.head())
print(rawData.info())
print(rawData.describe())

# Prediction features
predFeatures = ['Ship Mode', 'Order Date', 'Segment', 'Country', 'State', 'Region', 'Sub-Category', 'Category', 'Sales']

predDF = rawData[predFeatures].copy(deep = True)
predDF['logSales'] = np.log(predDF['Sales'])

# Plotting setup
colors = ["#2B4162", "#D7B377", "#606C38", "#ADB5BD", "#385F71", "#8F754F", "#283618", "#495057"]
sb.set_theme(style="whitegrid", palette = sb.color_palette(colors))

# Sales vs Shipping Mode
sb.boxplot(x = 'Ship Mode', y = 'logSales', data = predDF).set_title(label = "Shipping Mode vs Sales [Log Scale]")
plt.show()

# Sales vs Segment
sb.boxplot(x = 'Segment', y = 'logSales', data = predDF).set_title(label = "Customer Segment vs Sales [Log Scale]")
plt.show()

# Sales vs Category
sb.boxplot(x = 'Category', y = 'logSales', data = predDF).set_title(label = "Product Category vs Sales [Log Scale]")
plt.show()

# Forecast features
forFeatures = ['Order Date']
forResponse = ['Sales']



# Sales Forecast plot
sb.boxplot(x = 'Category', y = 'Sales', data = predDF).set_title(label = "Monthly Sales | 2015-2019")
plt.show()