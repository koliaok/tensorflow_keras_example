import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()# 전통 Tensorflow 학습 방식을 위해서 TF.2.0 버전에 Default인 Eager Excution을 Disable시킴
minist = tf.keras.datasets.mnist # mnist 데이터를 분ㄹ러우기 위해서 -> keras.datasets에서 불러옴

class Mnist(object):

    def load_data(self):
        """
        학습과 평가를 위한 데이터 Set을 불러옴
        기존 tf2.0 이하 버젼에서는 Mnist 를 불러와서 학습을 시킬때 이미지 데이터가 float32형태로 바로 Nomalization 되어 Output을 주지만,
        현재 tf2.0에서는 그렇지 않기 때문에 Data type과 Shape, Nomalization을 해야함
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = minist.load_data()
        self.w = []#weight 를 담을 List
        self.b = []#bais 를 담을 List
        self.x_test= self.x_test.astype(np.float32).reshape(self.x_test.shape[0], 784)/255.0
        self.x_train = self.x_train.astype(np.float32).reshape(self.x_train.shape[0], 784)/255.0

        def make_one_hot(label_data):
            output_data = []
            for y_label in label_data:
                zero_list = [0 for i in range(10)]
                zero_list[y_label] = 1
                output_data.append(zero_list)
            return np.array(output_data)

        self.y_train = make_one_hot(self.y_train)
        self.y_test = make_one_hot(self.y_test)

    def batch_func(self, batch_size, total_batch_size):
        """
        기존 Batch function을 대체 하기 위한 함수
        :param batch_size: 배치 사이즈
        :param total_batch_size: 다음 배치 사이즈를 구하기 위한 변수
        :return: 학습 데이터와 라벨 데이터를 뿌려줌
        """
        if total_batch_size==0:
            train_x = self.x_train[:batch_size]
            train_y = self.y_train[:batch_size]
            total_batch_size += batch_size
        else:
            train_x = self.x_train[total_batch_size:total_batch_size+batch_size]
            train_y = self.y_train[total_batch_size:total_batch_size+batch_size]
            total_batch_size += batch_size
        return train_x, train_y, total_batch_size

    def conv_layer(self, x, num_channel, num_output):
        """
        convolution layer define
        :param x: 입력된 이미지
        :param num_channel: 이미지 channel -> 1
        :param num_output: 10
        :return:
        """
        conv1_w = tf.compat.v1.Variable(tf.random.normal(dtype=tf.float32, shape=[4, 4, num_channel, 32],
                                                         stddev=0.1), name='conv1_w')
        conv1_b = tf.compat.v1.Variable(tf.random.normal(dtype=tf.float32, shape=[32]), name='conv1_b')

        conv2_w = tf.compat.v1.Variable(tf.random.normal(dtype=tf.float32, shape=[4, 4, 32, 64],
                                                         stddev=0.1), name='conv2_w')
        conv2_b = tf.compat.v1.Variable(tf.random.normal(dtype=tf.float32, shape=[64]), name='conv2_w')

        flatten_w = tf.compat.v1.Variable(tf.random.normal(dtype=tf.float32, shape=[64 *7 * 7  * 1, 1024],
                                                         stddev=0.1), name='flatten_w')
        flatten_b = tf.compat.v1.Variable(tf.random.normal(dtype=tf.float32, shape=[1024]), name='flatten_b')

        fully_connect_w = tf.compat.v1.Variable(tf.random.normal(dtype=tf.float32, shape=[1024, num_output],
                                                           stddev=0.1), name='fully_connect_w')
        fully_connect_b = tf.compat.v1.Variable(tf.random.normal(dtype=tf.float32, shape=[num_output]), name='fully_connect_b')

        #layer start

        conv1_layer = tf.compat.v1.nn.relu(tf.compat.v1.nn.conv2d(x, filter=conv1_w,
                                                                  strides=[1, 1, 1, 1], padding='SAME')+conv1_b)
        conv1_max_pooling = tf.compat.v1.nn.max_pool(conv1_layer, ksize=[1, 2, 2, 1],
                                                     strides=[1, 2, 2, 1], padding='SAME')

        conv2_layer = tf.compat.v1.nn.relu(tf.compat.v1.nn.conv2d(conv1_max_pooling, filter=conv2_w,
                                                                  strides=[1, 1, 1, 1], padding='SAME') + conv2_b)
        conv2_max_pooling = tf.compat.v1.nn.max_pool(conv2_layer, ksize=[1, 2, 2, 1],
                                                     strides=[1, 2, 2, 1], padding='SAME')



        conv2_max_pooling = tf.reshape(conv2_max_pooling, shape=[-1, 64 * 7 * 7 * 1])

        flatten_out  = tf.matmul(conv2_max_pooling, flatten_w) + flatten_b
        fully_connect_out = tf.matmul(flatten_out, fully_connect_w ) + fully_connect_b

        return fully_connect_out

    def start_learn_mnist(self):
        """
        실재 학습에 필요한 변수를 설정하고, 학습을 실행하는 부분
        """
        num_output = 10
        width = 28
        height = 28
        num_channel = 1

        num_input = width*height*num_channel
        learning_rate = 0.01
        epochs = 10
        batch_size = 128

        total_minist_data = len(self.x_train)
        n_batch_size = int(total_minist_data/batch_size)

        x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_input], name='input_x')
        y = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None, num_output], name='input_y')
        x_in = tf.reshape(x, shape=[-1, width, height, num_channel])

        logits = self.conv_layer(x_in, num_channel, num_output)

        loss = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        res_cnt = tf.equal(tf.argmax(logits,1), tf.argmax(y, 1))
        calculation = tf.reduce_mean(tf.cast(res_cnt, dtype=tf.float32), name='calculation')
        init = tf.compat.v1.initialize_all_variables()

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):
                epoch_accuracy = 0.0
                total_batch_size = 0
                for batch in range(n_batch_size):
                    x_batch, y_batch, total_batch_size = self.batch_func(batch_size, total_batch_size)
                    optimize_loss, _ = sess.run([loss, optimizer], feed_dict={x: x_batch, y: y_batch})
                    epoch_accuracy+=optimize_loss
                average = epoch_accuracy/n_batch_size
                print(f'Avge loss : {average}')
                print(f'Epoch : {epoch},,,  Average: {average}')

            accuracy = sess.run([calculation], feed_dict = {x: self.x_test, y: self.y_test})
            print(f'Test Accuracy : {accuracy}')


def main():
    minist = Mnist()
    minist.load_data()
    minist.start_learn_mnist()

if __name__ == '__main__':
    main()
