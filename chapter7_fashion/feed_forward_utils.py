from numba import jit
import numpy as np
import activation

"""
	순전파
	A1 = input *W1 + B1
	Z1 = Sigmoid(A1)
	A2 = Z1 * W2 + B2
	y = Sigmoid(A2)

	y 값을 바탕으로, 로그최대우도추정법을 활용하여 W1, B1, W2, B2 보정치를 계산하고 그 보정치를 리턴한다.
	"""
def feed_forward(input_data, target_data, W1, W2, B1, B2):
    delta = 1e-7 #log 무한대 발산 방지를 위해 추가한다.

    A1 = np.dot(input_data, W1) + B1
    Z1 = activation.Sigmoid(A1)
    A2 = np.dot(Z1, W2) + B2
    y = activation.Sigmoid(A2)
    # 로그 최대 우도 추정법을 사용한다.
    result = -np.sum(target_data * np.log(y + delta) + (1 - target_data) * np.log((1 - y) + delta))

    return result


@jit(nopython=True)
def feed_forward_partial_W1(input_data, target_data, W1, W2, B1, B2):
    h = 1e-7
    w1_partial = np.zeros(shape=(len(W1), len(W1[0])))
    for i in range(len(W1)):
        for j in range(len(W1[i])):
            modifiedPlusW1 = W1.copy()
            modifiedPlusW1[i][j] += modifiedPlusW1[i][j] + h

            modifiedMinusW1 = W1.copy()
            modifiedMinusW1[i][j] += modifiedMinusW1[i][j] - h

            ## feed_forward(input_data, target_data, modifiedPlusW1, W2, B1, B2) - feed_forward(input_data, target_data, modifiedMinusW1, W2, B1, B2)
            ## 위 함수 내용을 inline으로 옮겼다.
            # feed_forward 함수를 여기로 옮김.
            delta = 1e-7
            A1_plus = np.dot(input_data, modifiedPlusW1) + B1
            Z1_plus = 1 / (1+np.exp(-A1_plus))
            A2_plus = np.dot(Z1_plus, W2) + B2
            y = 1 / (1+np.exp(-A2_plus))
            plus_part = -np.sum(target_data * np.log(y + delta) + (1 - target_data) * np.log((1 - y) + delta))

            # feed_forward 함수를 여기로 옮김.
            delta = 1e-7
            A1_minus = np.dot(input_data, modifiedMinusW1) + B1
            Z1_minus = 1 / (1+np.exp(-A1_minus))
            A2_minus = np.dot(Z1_minus, W2) + B2
            y = 1 / (1+np.exp(-A2_minus))
            minus_part = -np.sum(target_data * np.log(y + delta) + (1 - target_data) * np.log((1 - y) + delta))

            upper_part = plus_part - minus_part
            partial_element = upper_part / (2*h)
            w1_partial[i][j] = partial_element

    return w1_partial