import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

parser = argparse.ArgumentParser(description='mode 입력')
parser.add_argument('--mode', default='SAE', help='Auto Encoder Mode')
arg_flag = parser.parse_args()

tf.compat.v1.disable_eager_execution()# 전통 Tensorflow 학습 방식을 위해서 TF.2.0 버전에 Default인 Eager Excution을 Disable시킴
minist = tf.keras.datasets.mnist # mnist 데이터를 분ㄹ러우기 위해서 -> keras.datasets에서 불러옴

class Mnist(object):
    # Function to display the images and labels
    # images should be in NHW or NHWC format
    def display_images(self, images, labels, count=0, one_hot=False):
        # if number of images to display is not provided, then display all the images
        if (count == 0):
            count = images.shape[0]

        idx_list = random.sample(range(len(labels)), count)
        for i in range(count):
            plt.subplot(4, 4, i + 1)
            plt.title(labels[i])
            plt.imshow(images[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

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

    def add_noise(self, x):
        return x + 0.5 * np.random.randn(x.shape[0], x.shape[1])

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

    def Stacked_Auto_Encoder(self, x, num_output):
        """
        Auto Encoder Hidden [512, 256, 256, 512]
        Flow  784(input) -> [512, 256, 256, 512](AutoEncoder) -> 784(output)
        """
        auto_encoder_list = [num_output, 512, 256, 512, num_output]
        layer = x
        for i in range(len(auto_encoder_list)-1):
            weight = tf.compat.v1.Variable(tf.random.normal(
                shape=[auto_encoder_list[i], auto_encoder_list[i+1]],
                dtype=tf.float32),
                name='w_'+str(i))

            bias = tf.compat.v1.Variable(tf.compat.v1.zeros(
                shape=[auto_encoder_list[i + 1]]),
                name='b_' + str(i))

            layer = tf.compat.v1.nn.sigmoid(tf.matmul(layer, weight) + bias)


        return layer

    def variational_auto_encoder(self, x, num_output):
        latent_variabel_list = 128
        input_list = [num_output, 512, 256]
        output_list = [latent_variabel_list, 256, 512, num_output]


        layer = x
        for i in range(len(input_list) - 1):
            weight = tf.compat.v1.get_variable(name='w_encoder' + str(i),
                shape = [input_list[i], input_list[i + 1]],
                initializer = tf.compat.v1.glorot_uniform_initializer())

            bias = tf.compat.v1.Variable(tf.zeros(shape=[input_list[i + 1]]), name='b_encoder' + str(i))

            layer = tf.compat.v1.nn.tanh(tf.matmul(layer, weight) + bias)

        w_encoder_mean = tf.compat.v1.get_variable(name='w_encoder_mean',
                                                   shape=[input_list[-1], latent_variabel_list],
                                                   initializer=tf.compat.v1.glorot_uniform_initializer())
        b_encoder_mean = tf.compat.v1.Variable(tf.zeros(shape=[latent_variabel_list]), name='b_encoder_mean')

        z_mean_variable = tf.matmul(layer, w_encoder_mean) + b_encoder_mean

        w_log_variable_encoder = tf.compat.v1.get_variable(name='w_log_variable_encoder',
                                                   shape=[input_list[-1], latent_variabel_list],
                                                   initializer=tf.compat.v1.glorot_uniform_initializer())
        b_log_variable_encoder = tf.compat.v1.Variable(tf.zeros(shape=[latent_variabel_list]), name='b_log_variable_encoder')
        z_log_variable = tf.matmul(layer, w_log_variable_encoder) + b_log_variable_encoder

        epsilon = tf.random.normal(shape=tf.shape(z_log_variable),
                                   mean=0,
                                   stddev=1.0,
                                   dtype=tf.float32,
                                   name='epsilon'
                                   )
        z = z_mean_variable + tf.exp(0.5*z_log_variable)*epsilon
        layer = z

        for i in range(len(output_list) - 1):
            weight = tf.compat.v1.get_variable(name='w_decider' + str(i),
                shape = [output_list[i], output_list[i + 1]],
                initializer = tf.compat.v1.glorot_uniform_initializer())
            bias = tf.compat.v1.Variable(tf.compat.v1.zeros(
                shape=[output_list[i + 1]]),
                name='b_decoder' + str(i))

            if i == len(output_list) - 2:
                layer = tf.compat.v1.nn.sigmoid(tf.matmul(layer, weight) + bias)
            else:
                layer = tf.compat.v1.nn.tanh(tf.matmul(layer, weight) + bias)

        return layer, z_log_variable, z_mean_variable


    def start_learn_mnist(self, mode):
        """
        실재 학습에 필요한 변수를 설정하고, 학습을 실행하는 부분
        """

        width = 28
        height = 28
        num_output = width*height
        learning_rate = 0.001
        epochs = 30
        batch_size = 100

        total_minist_data = len(self.x_train)
        n_batch_size = int(total_minist_data/batch_size)

        x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_output], name='input_x')

        if mode=='VAE':
            logits, z_log_var, z_mean = self.variational_auto_encoder(x, num_output)
            rec_loss = -tf.reduce_sum(x * tf.math.log(1e-10 + logits) + (1 - x) * tf.math.log(1e-10 + 1 - logits), 1)
            reg_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1)
            loss = tf.reduce_mean(rec_loss + reg_loss)
        else:
            logits = self.Stacked_Auto_Encoder(x, num_output)
            loss = tf.compat.v1.losses.mean_squared_error(labels=x, predictions=logits)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        init = tf.compat.v1.initialize_all_variables()

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):
                epoch_accuracy = 0.0
                total_batch_size = 0
                for batch in range(n_batch_size):
                    x_batch, _, total_batch_size = self.batch_func(batch_size, total_batch_size)
                    if mode=='DAE':
                        x_batch = self.add_noise(x_batch)
                    optimize_loss, _ = sess.run([loss, optimizer], feed_dict={x: x_batch})
                    epoch_accuracy+=optimize_loss
                average = epoch_accuracy/n_batch_size
                print(f'Avge loss : {average}')
                print(f'Epoch : {epoch},,,  Average: {average}')

            x_image, y_label, total_batch_size = self.batch_func(4, 0)

            y_labels = [np.argmax(res) for res in y_label]
            predict_image = sess.run(logits, feed_dict={x:x_image})
            self.display_images(x_image.reshape(-1, 28, 28), y_labels)
            self.display_images(predict_image.reshape(-1, 28, 28), y_labels)

            if mode == 'DAE':
                x_noise = self.add_noise(x_image)
                predict_image = sess.run(logits, feed_dict={x: x_noise})
                self.display_images(x_noise.reshape(-1, 28, 28), y_labels)
                self.display_images(predict_image.reshape(-1, 28, 28), y_labels)


def main():
    minist = Mnist()
    minist.load_data()
    minist.start_learn_mnist(arg_flag.mode)

if __name__ == '__main__':
    main()
