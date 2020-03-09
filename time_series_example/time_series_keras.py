import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_eager_execution()

# data define
csv_file = 'airline_file.csv'
datafram = pd.read_csv(csv_file, usecols=[1], header=0)
value_df = datafram.values.astype(np.float32)

minmax_model = MinMaxScaler(feature_range=(0, 1))
nomalized_dataset = minmax_model.fit_transform(value_df)
train, test = sklearn.model_selection.train_test_split(nomalized_dataset, train_size=0.77, shuffle=False)


class TimeSeriesTensorflow(object):

    def get_batch(self, start_cnt, batch_size, window_size, output_number, source_data):
        x_train_data = []
        y_train_data = []
        for batch in range(batch_size):
            x_train_data.append(source_data[batch + start_cnt: batch + start_cnt + window_size])
            y_train_data.append(
                source_data[batch + start_cnt + window_size: batch + start_cnt + window_size + output_number])
        start_cnt += batch_size
        return np.array(x_train_data), np.array(y_train_data), start_cnt

    def train_sries_data(self):

        #tensorf value define
        epoch = 100
        train_batch_size = len(train) - 1
        test_batch_size = len(test) - 1
        rnn_parameter = 4
        window_size = 1
        output_number = 1
        learning_rate = 0.1
        start_cnt=0

        #Dynamic하게 Series data Batch를 만들기 위해서 계산한 부분
        train_iteration_val = window_size+train_batch_size+output_number-1
        train_data_size = int((len(train)-train_iteration_val+train_batch_size)//train_batch_size)

        test_iteration_val = window_size + test_batch_size + output_number - 1
        test_data_size = int((len(test)-test_iteration_val+test_batch_size)//test_batch_size)

        feed_x_train, feed_y_train, start_cnt = self.get_batch(0, train_batch_size, window_size,
                                                               output_number, train)
        feed_y_train = feed_y_train.reshape(feed_y_train.shape[0], feed_y_train.shape[1])

        feed_x_test, feed_y_test, start_cnt = self.get_batch(0, test_batch_size, window_size,
                                                               output_number, test)

        # model define
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.SimpleRNN(units=rnn_parameter, input_shape=(feed_x_train.shape[1], feed_x_train.shape[2])))
        model.add(tf.keras.layers.Dense(1))
        model.summary()

        # loss and optimizer define
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())

        # model learning
        model.fit(feed_x_train, feed_y_train, batch_size=1, epochs=100)

        # model predict
        res_logits = model.predict(feed_x_test)

        predict_graph_list = res_logits
        feed_y_test = feed_y_test.reshape(-1, 1)
        predict_data = minmax_model.inverse_transform(predict_graph_list)
        testPredictPlot = minmax_model.inverse_transform(feed_y_test)
        plt.plot(testPredictPlot, label='Original TeST')
        plt.plot(predict_data, label='Predict Test')
        plt.legend()
        plt.xlabel('Timesteps')
        plt.ylabel('Total Passengers')
        plt.show()

    def rnn_model(self):
        pass


def main():
    time_series_tensorflow = TimeSeriesTensorflow()
    time_series_tensorflow.train_sries_data()


if __name__ == '__main__':
    main()