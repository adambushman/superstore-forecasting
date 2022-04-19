import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

rawData = pd.read_csv('train-store-data.csv')
print(rawData.head())
print(rawData.describe())

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

sales_forcasting_df = rawData['Sales']
print(sales_forcasting_df.head())
sales_forcasting_df.plot()
plt.show()


#Successfully linked editor and GitHub! (Adam)
#I'm in! (Zach)