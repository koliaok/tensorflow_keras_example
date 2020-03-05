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
        :return:
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = minist.load_data()
        self.w = []#weight 를 담을 List
        self.b = []#bais 를 담을 List
        self.x_test= self.x_test.astype(np.float32).reshape(self.x_test.shape[0], 784)/255.0
        self.x_train = self.x_train.astype(np.float32).reshape(self.x_train.shape[0], 784)/255.0
        self.y_train =np.eye(10)[self.y_train]
        self.y_test = np.eye(10)[self.y_test]

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

    def MLP(self, x, number_layer, number_nueral, num_input, num_output):
        """
        특정 입력값을 받아드려서 Layer를 만듬
        :param x: 학습 데이터
        :param number_layer: 학습 네트워크 Layer 층
        :param number_nueral: 파라미터 개수
        :param num_input: 입력 차원수
        :param num_output: 출력 차원수
        :return: Layer 마지막 Logit 값
        """
        layer = x
        for number in range(number_layer):
            if number == number_layer - 1:
                self.w.append(tf.Variable(tf.random.normal([number_nueral[-1], num_output])))
                self.b.append(tf.Variable(tf.random.normal([num_output])))
                break
            self.w.append(tf.Variable(
                tf.random.normal([num_input if number == 0 else number_nueral[number - 1], number_nueral[number]])))
            self.b.append(tf.Variable(tf.random.normal([number_nueral[number]]), name=f'b_{str(number)}'))

        for num, (weight, bias) in enumerate(zip(self.w, self.b)):
            if num == len(self.w) - 1:
                layer = tf.matmul(layer, weight) + bias
            else:
                layer = tf.nn.relu(tf.matmul(layer, weight) + bias)

        return layer

    def start_learn_minist(self):
        """
        실재 학습에 필요한 변수를 설정하고, 학습을 실행하는 부분
        :return:
        """

        num_output = 10
        num_input = 784
        number_layer = 3
        number_nueral = [100, 100, 100]
        learning_rate = 0.01
        epochs = 50
        batch_size = 100

        total_minist_data = len(self.x_train)
        n_batch_size = int(total_minist_data/batch_size)

        x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_input])
        y = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, num_output])


        logits = self.MLP(x, number_layer, number_nueral, num_input, num_output)

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
    minist.start_learn_minist()

if __name__ == '__main__':
    main()
