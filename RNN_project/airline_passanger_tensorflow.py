import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
csv_file = 'international-airline-passengers.csv'

datafram = pd.read_csv(csv_file, usecols=[1], header=0)
value_df = datafram.values.astype(np.float32)
minmax_model = MinMaxScaler(feature_range=(0,1))
nomalized_dataset = minmax_model.fit_transform(value_df)
train, test = sklearn.model_selection.train_test_split(nomalized_dataset, train_size=0.67)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(train, test, test_size=0.1)
print(train)
class TimeSeriesTensorflow(object):

    def get_batch(self, batch_size, window):



