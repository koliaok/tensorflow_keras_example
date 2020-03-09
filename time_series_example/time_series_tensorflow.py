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

    def rnn_net(self, rnn_parameter, x_seris_data, output_number, mode='RNN'):

        if mode=="RNN":
            cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=rnn_parameter)
        else:
            cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=rnn_parameter)
        rnn_outputs, final_state = tf.compat.v1.nn.static_rnn(cell, x_seris_data, dtype=tf.float32)
        W = tf.compat.v1.Variable(tf.random.normal([rnn_parameter, output_number]), dtype=tf.float32, name='rnn_w')
        B = tf.compat.v1.Variable(tf.random.normal([output_number]), dtype=tf.float32, name='rnn_b')
        return [tf.compat.v1.matmul(rnn_res, W) + B for rnn_res in rnn_outputs]


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

        #Dynamic하게 Series data Batch를 만들기 위해서 계산한 부분
        train_iteration_val = window_size+train_batch_size+output_number-1
        train_data_size = int((len(train)-train_iteration_val+train_batch_size)//train_batch_size)

        test_iteration_val = window_size + test_batch_size + output_number - 1
        test_data_size = int((len(test)-test_iteration_val+test_batch_size)//test_batch_size)



        x_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, window_size, output_number])
        y_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, window_size, output_number])
        x_seris_data = tf.unstack(x_input, axis=1)
        y_label = tf.unstack(y_input, num=window_size, axis=1)

        # model define
        logits = self.rnn_net(rnn_parameter, x_seris_data, output_number, mode="RNN")

        # loss and optimization
        loss = tf.reduce_mean([tf.compat.v1.losses.mean_squared_error(labels=y, predictions=x)
                               for y, x in zip(y_label, logits)])
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss=loss)

        # tensor excution flow
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for ep in range(epoch):
                average_loss = 0
                start_cnt = 0
                print(f'epoch : {ep}')
                for excution_cnt in range(train_data_size):
                    feed_x_train, feed_y_train, start_cnt = self.get_batch(start_cnt, train_batch_size, window_size,
                                                                           output_number, train)

                    training_loss, _ = sess.run([loss, optimizer],feed_dict={x_input: feed_x_train, y_input: feed_y_train})
                    average_loss += training_loss
                print(f'Average loss : {average_loss / train_data_size}')

            # evaluation flow
            start_cnt = 0
            for excution_cnt in range(test_data_size):
                feed_x_test, feed_y_test, start_cnt = self.get_batch(start_cnt, test_batch_size, window_size,
                                                                       output_number, test)
                res_logits,_ = sess.run([logits, loss], feed_dict={x_input: feed_x_test, y_input: feed_y_test})



        predict_graph_list = res_logits[0]
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