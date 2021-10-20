from datetime import datetime
import pandas as pd
import numpy as np
import trainer
import time


datetimenow = str(datetime.now())
f = open("train_result/"+datetimenow+".txt", "w")


df_train = pd.read_csv('data/fashion-mnist_train.csv')
df_test = pd.read_csv('data/fashion-mnist_test.csv')

data_train = np.array(df_train, dtype = np.float32)
data_test = np.array(df_test, dtype = np.float32)

# input nodes, hidden nodes, output nodes, learning rate
my_model = trainer.my_NN01(784,40,10,0.05)

# len(data_train)=60000개
for step in range(len(data_train)):
    # 원-핫-인코딩 형태로 정규화한다.
    input_data = ((data_train[step, 1:]/255.0)*0.99) + 0.01
    input_data = np.float64(input_data)

    # 출력 변수를 은닉층 개수인 10개만큼 생성하고 0.01로 초기화한다
    target_data = np.zeros(10) + 0.01
    # 정답에 해당되는 출력 변수는 0.99로 설정한다.
    target_data[int(data_train[step,0])] = 0.99
    start_time = time.time();
    my_model.train(input_data, target_data)
    end_time = time.time();

    print(step, '번째 훈련 중.. 비용은 ', my_model.cost())
    print('걸린 시간: ', end_time-start_time, "초")
    if step % 100 == 0 :
        accurate_rate = my_model.accuracy(data_test)
        f.write('{0} 번째 훈련 중... 정확도: {1}%'.format(step, accurate_rate))
        f.write('\n')


accurate_rate = my_model.accuracy(data_test)
f.write('최종 정확도: {1}%'.format(accurate_rate))
f.write('\n')
f.close()
