import numpy as np
import activation

class my_NN01:
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		
		# 아래 input_nodes, hidden_nodes, output_nodes은 숫자가 들어간다 (ex:3)
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		# W1, B1, W2, B2에는 numpy.ndarray 자료형이 들어간다.
		# W1, W2는 수렴값을 빠르게 찾기 위해 입력층의 노드 수의 절반의 제곱근으로 나눈다.
		self.W1 = np.random.rand(self.input_nodes, self.hidden_nodes) / np.sqrt(self.input_nodes/2)
		self.B1 = np.random.rand(self.hidden_nodes)

		self.W2 = np.random.rand(self.hidden_nodes, self.output_nodes) / np.sqrt(self.hidden_nodes/2)
		self.B2 = np.random.rand(self.output_nodes)

		self.learning_rate = learning_rate

	"""
	순전파
	A1 = input *W1 + B1
	Z1 = Sigmoid(A1)
	A2 = Z1 * W2 + B2
	y = Sigmoid(A2)

	y 값을 바탕으로, 로그최대우도추정법을 활용하여 W1, B1, W2, B2 보정치를 계산하고 그 보정치를 리턴한다.
	"""
	def feed_forward(self, W1, W2, B1, B2):
		delta = 1e-7 #log 무한대 발산 방지를 위해 추가한다.

		A1 = np.dot(self.input_data, W1) + B1
		Z1 = activation.Sigmoid(A1)
		A2 = np.dot(Z1, W2) + B2
		y = activation.Sigmoid(A2)

		# 로그 최대 우도 추정법을 사용한다.
		return -np.sum(self.target_data*np.log(y+delta) + (1-self.target_data)*np.log((1-y) + delta))

	def feed_forward_partial_W1(self):
		h = 1e-7

		upper_part = (self.feed_forward(self.W1+h, self.W2, self.B1, self.B2) - self.feed_forward(self.W1-h, self.W2, self.B1, self.B2))
		return upper_part / (2*h)

	def feed_forward_partial_W2(self):
		h = 1e-7

		upper_part = (self.feed_forward(self.W1, self.W2+h, self.B1, self.B2) - self.feed_forward(self.W1, self.W2-h, self.B1, self.B2))
		return upper_part / (2*h)

	def feed_forward_partial_B1(self):
		h = 1e-7

		upper_part = (self.feed_forward(self.W1, self.W2, self.B1+h, self.B2) - self.feed_forward(self.W1, self.W2, self.B1-h, self.B2))
		return upper_part / (2*h)

	def feed_forward_partial_B2(self):
		h = 1e-7

		upper_part = (self.feed_forward(self.W1, self.W2, self.B1, self.B2+h) - self.feed_forward(self.W1, self.W2, self.B1, self.B2-h))
		return upper_part / (2*h)
	


	def cost(self):
		delta = 1e-7 # log 무한대 발산 방지를 위해 추가한다.

		A1 = np.dot(self.input_data, self.W1) + self.B1
		Z1 = activation.Sigmoid(A1)
		A2 = np.dot(Z1, self.W2) + self.B2
		y = activation.Sigmoid(A2)

		# 로그 최대 우도 추정법을 사용한다.
		cost_val =  -np.sum(self.target_data*np.log(y+delta) + (1-self.target_data)*np.log((1-y) + delta))
		return cost_val


	def train(self, input_data, target_data):
		self.input_data = input_data
		self.target_data = target_data
		
		# 순전파. y값을 계산하고 W1,W2,B1,B2 보정치를 계산한 후 그 보정치를 반환하는 함수를 편미분한다.
		# W1에 대해 편미분한다
		print('W1 편미분 전: ', self.W1)
		print('w1 편미분 차감값:',self.learning_rate * self.feed_forward_partial_W1())
		self.W1 -= self.learning_rate * self.feed_forward_partial_W1()
		print('W1 편미분 후: ', self.W1)
		# B1에 대해 편미분한다
		print('B1 편미분 전: ', self.B1)
		print('B1 편미분 차감값: ', self.learning_rate * self.feed_forward_partial_B1())
		self.B1 -= self.learning_rate * self.feed_forward_partial_B1()
		print('B1 편미분 후: ', self.B1)
		# W2에 대해 편미분한다
		print('W2 편미분 전: ', self.W2)
		print('W2 편미분 차감값: ', self.learning_rate * self.feed_forward_partial_W2())
		self.W2 -= self.learning_rate * self.feed_forward_partial_W2()
		print('W2 편미분 후: ', self.W2)

		# B2에 대해 편미분한다
		print('B2 편미분 전: ', self.B2)
		print('B2 편미분 차감값: ',self.learning_rate * self.feed_forward_partial_B2())
		self.B2 -= self.learning_rate * self.feed_forward_partial_B2()
		print('B2 편미분 후: ',self.B2)

	def predict(self, input_data):
		A1 = np.dot(self.input_data, self.W1) + self.B1
		Z1 = activation.Sigmoid(A1)
		A2 = np.dot(Z1, self.W2) + self.B2
		y = activation.Sigmoid(A2)
		predicted_num = np.argmax(y)
		return predicted_num

	def accuracy(self, test_data):
		matched_list = []
		not_matched_list = []

		for index in range(len(test_data)):
			label = int(test_data[index,0])
		
			# 데이터를 원-핫-인코딩 형태로 변환하기 위해 정규화한다.
			data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01
			# predict 데이터는 2차원 벡터 형태로 연산하므로, 1차원 데이터인 data를 2차원으로 변환한다.
			predicted_num = self.predict(np.array(data, ndmin = 2))

			if label == predicted_num:
				matched_list.append(index)
			else:
				not_matched_list.append(index)

		print('정확도: ', 100 *(len(matched_list) / len(test_data)), '%')
		return matched_list, not_matched_list



