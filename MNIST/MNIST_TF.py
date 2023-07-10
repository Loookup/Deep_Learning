import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. MNIST 데이터셋 임포트
mnist = tf.keras.datasets.mnist
(x_train, t_train), (x_test, t_test) = mnist.load_data()

# 2. 데이터 전처리
x_train, x_test = x_train/255.0, x_test/255.0

# 3. 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
# model.fit(x_train, y_train, epochs=5)
#
# # 6. 정확도 평가
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print('테스트 정확도:', test_acc)


iters_num = 1000
train_size = x_train[0].shape
batch_size = 100
learning_rate = 0.1


train_loss_list = []
train_acc_list = []


for i in range(10):
    print(i + 1, 'th iter')
    # batch_mask = np.random.choice(train_size, batch_size)
    # x_batch = x_train[batch_mask]
    # t_batch = t_train[batch_mask]
    # model.fit(x_batch, t_batch, epochs=1)

    model.fit(x_train, t_train, epochs=1)
    train_loss_list.append(model.evaluate(x_test, t_test)[0])
    train_acc_list.append(model.evaluate(x_test, t_test)[1])


x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='loss')
plt.plot(x, train_acc_list, label='acc')
plt.legend()
plt.show()



# def get_data():
#     # Import Datasets
#     mnist = tf.keras.datasets.mnist
#     (x_train, t_train), (x_test, t_test) = mnist.load_data()
#
#     # Flatten
#     x_train, x_test = x_train.reshape(60000, 784), x_test.reshape(10000, 784)
#
#     # Normalize
#     x_train, x_test = x_train / 255.0, x_test / 255.0
#
#     return x_test, t_test
#
# def sigmoid(X):
#     return 1/(1+np.exp(-X))
#
# def identity_Function(X):
#     return X
#
# def softmax(X):
#     # Vulnerable to Overflow
#     exp_X = np.exp(X)
#     sum_exp_X = np.sum(exp_X)
#     Y = exp_X / sum_exp_X
#
#     return Y
#
# def init_network():
#
#     with open("sample_weight.pkl", 'rb') as f:
#         network = pickle.load(f)
#
#     return network
#
# def predict(network, x):
#
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']
#
#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y = softmax(a3)
#
#     return y
#
#
# x, t = get_data()
# network = init_network()
#
#
# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)
#     if p == t[i]:
#         accuracy_cnt += 1
#
#
# print("Acc :" + str(float(accuracy_cnt) / len(x)))



# Shape
# print(x_train.shape) # (60000, 28, 28)
# print(t_train.shape) # (60000, )
# print(x_test.shape) # (10000, 28, 28)
# print(t_test.shape) # (10000, )
#
# # Index
# label = t_train[0]
# print(label)

# Image Check
# cv2.imshow("Window", x_train[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

