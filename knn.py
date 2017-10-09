import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  
import csv
from sklearn.preprocessing import StandardScaler


# # read flash.dat to a list of lists
# datContent = [i.strip().split() for i in open("train.dat").readlines()]

# # write it as a new CSV file
# with open("train.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(datContent)
# def your_func(row):
#     return row['Sentiments'] / row['Review']

# columns_to_keep = ['Sentiments', 'Review']
# dataframe = pd.read_csv("train.csv", usecols=columns_to_keep)
# dataframe['new_column'] = dataframe.apply(your_func, axis=1)

# print dataframe
# df = pd.read_fwf('train.dat', header=None, 
#         widths=[2, int(1e15)], names=['Sentiments', 'Review'])

# print(df)
# scaler = StandardScaler()
# scaled_features = scaler.transform(df.drop('Sentiments',axis=1))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# read in the dataset
df = pd.read_csv(
    filepath_or_buffer='train.dat', 
    header=None, 
    #sep=','
    )
# extract the vectors from the Pandas data file
X = df.iloc[:,1:].values

# standardise the data
X_std = StandardScaler().fit_transform(X)
print(X_std)
