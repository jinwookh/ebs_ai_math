import pandas as pd
import numpy as np
import trainer

df_train = pd.read_csv('data/fashion-mnist_train.csv')
df_test = pd.read_csv('data/fashion-mnist_test.csv')

data_train = np.array(df_train, dtype = np.float32)
data_test = np.array(df_test, dtype = np.float32)

my_model = trainer.my_NN01(784, 100,10,0.01)

# for step in range(len(data_train)):
# 원-핫-인코딩 형태로 정규화한다.
input_data = ((data_train[0, 1:]/255.0)*0.99) + 0.01

# 출력 변수를 은닉층 개수인 10개만큼 생성하고 0.01로 초기화한다
target_data = np.zeros(10) + 0.01
# 정답에 해당되는 출력 변수는 0.99로 설정한다.
target_data[int(data_train[0,0])] = 0.99

my_model.train(input_data, target_data)

print( '번째 훈련 중.. 비용은 ', my_model.cost())

# my_model.accuracy(data_test)