import tensorflow as tf
import pandas as pd
import numpy as np

df_train = pd.read_csv('data/fashion-mnist_train.csv')
df_test = pd.read_csv('data/fashion-mnist_test.csv')

data_train = np.array(df_train)
data_test = np.array(df_test)

x_train = data_train[:,1:]
y_train = data_train[:,0]

x_test = data_test[:,1:]
y_test = data_test[:,0]

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)


print(type(x_train)) # numpy.ndarray


print("x_train:%s y_train:%s x_test:%s y_test:%s" %(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
# x_train: (60000, 28, 28) y_train:(60000,) x_test:(10000, 28, 28), y_test: (10000,)
# x has pixel data, y has label data.

import matplotlib.pyplot as plt

for y in range(28):
	for x in range(28):
		print("%4s" %x_train[0][y][x], end = '')
	print()
# show pixel value of one picture, which is has 28x28 size datail


class_names = [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape(60000,784), x_test.reshape(10000, 784)

print("after resize: x_train:%s  x_test:%s " %(x_train.shape, x_test.shape))
# after resize: x_train:(60000, 784)  x_test:(10000, 784)



model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=(784,)),
	tf.keras.layers.Dense(128, activation = 'relu'),
	tf.keras.layers.Dense(10, activation = 'softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

p_test = model.predict(x_test)
print('p_test[0]: ', p_test[0])
# ex) p_test[0]:  [2.1941682e-06 3.0824889e-08 4.7764496e-07 7.5205577e-08 1.4850874e-06  2.3567384e-02 8.1755961e-06 1.6137521e-01 1.8850867e-05 8.1502610e-01]


print ('p_test[0]: ', np.argmax(p_test[0]), class_names[np.argmax(p_test[0])], 'y_test[0]: ', y_test[0], class_names[y_test[0]])
# ex) p_test[0]:  9 Ankle boot y_test[0]:  9 Ankle boot
# numpy.argmax returns the index of the maximum value of array, which would be the highest probable garment index in this case

x_test = x_test.reshape(10000, 28, 28)

cnt_wrong = 0
p_wrong = []
for i in range(10000):
	if np.argmax(p_test[i]) != y_test[i]:
		p_wrong.append(i)
		cnt_wrong += 1

print('cnt_wrong: ', cnt_wrong)
print('predicted wrong 10: ', p_wrong[:10])

plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(x_test[p_wrong[i]], cmap=plt.cm.binary)
	plt.xlabel("%s: x-%s o-%s" %(
		p_wrong[i], class_names[np.argmax(p_test[p_wrong[i]])],
		class_names[y_test[p_wrong[i]]]))
plt.show()
# shows mispredicted case - format is index: x-predicted(wrong) o-answer
