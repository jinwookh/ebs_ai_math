import numpy as np

def Step(x):
	# numpy 배열을 인자로 받아, 0보다 큰 원소들은 1로, 0보다 작은 원소들은 0으로 초기화한 배열을 리턴한다.
	return np.array(x > 0, dtype = np.int)

def Sigmoid(x):
	# 로지스틱 함수 y=1/1+e^-x 를 표현한 코드
	return 1 / (1+np.exp(-x))

def ReLU(x):
	return np.maximum(0,x)


"""
Softmax는 확률의 기본 성질을 만족한다.
확률의 기본 성질
(1) 임의의 사건 A에 대하여 0 <= P(A) <= 1
   -> 0 <= Softmax 결과값 <= 1
(2) 반드시 일어나는 사건 S에 대하여 P(S) = 1
   -> 밑 Softmax 함수는 배열을 인자로 받아서 배열을 반환한다.
   -> 반환한 배열의 원소들을 모두 합하면 1이다.
"""
def Softmax(a):
	c = np.max(a)
	# e의 지수승이 결과가 나오기 때문에, 결과가 int64가 저장할 수 있는 최대치를 넘을 수 있다.
	# overflow를 방지하기 위해, 각 배열의 원소마다 최댓값을 뺀다.
	exp_a = np.exp(a-c)
	sum_exp_a = np.sum(exp_a)
	return exp_a / sum_exp_a

"""
overflow에 대한 방어로직이 없이 Softmax 방식으로 계산한다.
결과는 위 Softmax 함수와 같다.
위 Softmax 함수의 계산 결과와 왜 같은지는 한 번 증명해 볼 것.
"""
def SoftmaxWithoutOverflowPrevention(a):
	exp_a = np.exp(a)
	sum_exp_a = np.sum(exp_a)

	return exp_a / sum_exp_a



