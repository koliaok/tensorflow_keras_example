from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import tensorflow as tf

"""
Tensorflow 기본 학습 예제를 가져와서 Keras로 Mnist로 학습하는 코드 작성
"""
number_nueral = [100, 100]
num_output = 10
learning_rate = 0.01
batch_size = 100

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential() # 학습 Model 선언
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # mnist의 입력 데이터 Shape(batch, 28, 28) float32 이미지 데이
model.add(tf.keras.layers.Dense(number_nueral[0], activation=tf.keras.activations.relu))#Feed Foward 네트워크 와 Activation함수
model.add(tf.keras.layers.Dense(number_nueral[1], activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(num_output, activation=tf.keras.activations.softmax)) #마지막에 Softmax를 사용해 Label에 가장 가까운 결과를 출력
model.summary() # 학습 Model 요약

#tensorflow와 다르게 Optimizer와 Loss, 평가 Metrics까지 한번에 정의 및 적용
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#fit은 학습데이터와 Epochs, Batch_size 를 넣고 계산
model.fit(x=x_train, y=y_train, epochs=10, batch_size=batch_size)
test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=2, batch_size=batch_size)

