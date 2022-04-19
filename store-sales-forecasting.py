import os
import pandas as pd

# Change to your local directory
#os.chdir('C:/Users/Adam Bushman/Downloads') 

rawData = pd.read_csv('train-store-data.csv')
print(rawData.head())
print(rawData.info())

#Successfully linked editor and GitHub! (Adam)