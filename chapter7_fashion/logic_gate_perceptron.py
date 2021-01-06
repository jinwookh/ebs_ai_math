def AND(x1, x2):
	#파라미터 값
	#w1, w2는 가중치
	w1, w2, threshold = 0.2, 0.2, 0.3
	temp = w1 * x1 + w2 * x2
	if temp <= threshold:
		return 0
	elif temp > threshold:
		return 1


def OR(x1, x2):
	w1,w2, threshold = 0.3, 0.3, 0.2
	temp = w1 * x1 + w2 * x2
	if temp <= threshold:
		return 0
	else:
		return 1

def NAND(x1, x2):
	w1, w2, threshold = -0.2, -0.2, -0.3
	temp = w1 * x1 + w2 * x2
	if temp <= threshold:
		return 0
	else:
		return 1

def XOR(x1, x2):
	h1 = NAND(x1, x2)
	h2 = OR(x1, x2)
	Y = AND(h1, h2)
	return Y

