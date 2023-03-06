import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel(3)
from analyze import testThreshold, testPredict

import numpy as np
import math
import csv

#x_dividers code:
#[1.1,1.2,1.5, 2, 2.6, 3.5, 5, 10], [1.1,1.5, 2, 3.5] = 1
#[1.1,1.5, 2, 3.5, 10] = 2
#[1.1, 1.7, 2.3, 3, 4, 8] = 3
#[1.1,1.5, 2, 3.5, 10], [1.1, 3.5] for y = 4
#[1.1,1.2,1.5, 2, 2.6, 3.5, 5, 10], [1.1, 3.5] = 5
#1 but first layer = 5 instead of 12, dropout = 0.1 not 0.15, relu instead of softmax for finish = 6
#1 but learning rate 0.18 instead of 0.15 = 7
#4.2 is 1.4 but with [1.1, 2, 3.5] for y
#1.8 = [1.1,1.2,1.5, 2, 2.6, 3.5, 5, 10], [1.1,1.5, 2, 3.3, 5]
#5.1 = 1.1 except 5.1 is sorted *** (1.11) [(1.06)]
#5.115 = 5.1 except with 20,000 epochs (1.09/1.15) [1.12] [(1.10)]
#3.11 = 3.1 except 1.11 is sorted (1.05)
#4.11 = 4.1 = 3.1 except 4.11 is sorted ** (0.96)
#6.11 = 6.1 except 6.11 is sorted *_* (1.01) [(0.99)]
#6.115 is 6.11 but with 10x the epochs ** (1.05) [(1.09)]
#6.116 is 6.115 but with 20,000 epochs * (1.07) [(1.00)]
#5.12 = 5.1 but with [1.1, 2, 3] (for keyboard quickness) (0.96)
#4.12 = 4.11 but with [1.1, 2, 3] (for keyboard quickness) ** (0.98)
#6.12 = same as above (0.98)
#5.13 = 5.11 but with #[1.1,1.5, 2, 3.5, 7] as x-divider ** (0.90)
#4.13 same as above --
#5.215 = [1.1,1.24,1.45, 1.85, 2.5, 3.5, 5.3, 10], [1.2,1.5, 2, 3.5] sorted, 10k epochs [1.03] (0.99/1.09)
#6.215 same as above (/1.00) []
#4.215


x_dividers = [1.1,1.2,1.5, 2, 2.6, 3.5, 5, 10]
# x_dividers = [1.1,1.24,1.45, 1.85, 2.5, 3.5, 5.3, 10]
X_LEN = len(x_dividers) + 1
x_code = 115


y_dividers = [1.1,1.5, 2, 3.5]
# y_dividers = [1.2,1.5,2,3.5]
# y_dividers = [1.1, 2, 3]
Y_LEN = len(y_dividers) + 1

NUM_BEH = 5

testing_model = f"behind{NUM_BEH}_model{x_code}"

def transform_num(num):
    for i, x in enumerate(x_dividers):
        if num < x:
            return i
    return X_LEN - 1

def transform_num_y(num):
    # print(f"Num: {num}")
    val = [0] * Y_LEN
    for i, y in enumerate(y_dividers):
        if num < y:
            val[i] = 1
            # print(f"Transformed val: {val}")
            return val
    val[Y_LEN-1] = 1
    # print(f"Transformed val: {val}")
    return val

def y_to_num(y):
    if y == 0:
        return 1
    else:
        return y_dividers[y-1]

def thres_to_y(thres):
    if thres < 1.1:
        return 0
    if thres < 1.5:
        return 1
    if thres < 2:
        return 2
    if thres < 3:
        return 3
    else:
        return 4

def logReg1(x): #[1.1,1.24,1.45, 1.85, 2.5, 3.5, 5.3, 10]
        a = -9.41995
        b = 17.255
        c = 0.867079
        d = -0.20295
        g = 0.185033
        return a*math.pow(math.log(g*x+c),d) + b


def getMax(probs): #list is of pairs of y_divider indicator and probability that one will occur
    sumProbs = list(map(lambda i:sum(probs[i:]), range(Y_LEN)))
    def getEV(pair):
        evWin = pair[1] * (y_to_num(pair[0]) - 1)
        evLoss = (1 - pair[1]) * -1
        return evWin + evLoss
    return max(list(zip(range(Y_LEN), sumProbs)), key=getEV)

def testModel():
    x_test = []
    y_test = []
    y_real = []
    with open(f'behind{NUM_BEH}.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            added_list = list(transform_num(float(x)) for x in row[:NUM_BEH])
            added_list.sort()
            x_test.append(added_list)
            y_real.append(float(row[-1]))
            y_test.append(transform_num_y(float(row[-1])))

    # with open(f'behind{NUM_BEH}Play2.csv', newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for row in reader:
    #         added_list = list(transform_num(float(x)) for x in row[:NUM_BEH])
    #         added_list.sort()
    #         x_test.append(added_list)
    #         y_real.append(float(row[-1]))
    #         y_test.append(transform_num_y(float(row[-1])))

    # normalize the data
    mins = [0.0] * NUM_BEH
    maxes = [0.0] * NUM_BEH
    for i in range(0, NUM_BEH):
        mins[i] = min([x[i] for x in x_test])
        maxes[i] = max([x[i] for x in x_test])

    print(f"mins: {mins}, maxes: {maxes}")

    x_norm = [[(x[i] - mins[i]) / (maxes[i] - mins[i]) * 1.0 + 0.0 for i in range(0, NUM_BEH)] for x in x_test]

    model = tf.keras.models.load_model(testing_model)

    y_predict = [getMax(y)[0]\
                 for y in model.predict(np.matrix(x_norm))]
    y_predict2 = [max(list(zip(range(Y_LEN), y)), key=lambda x:x[1])[0]\
                 for y in model.predict(np.matrix(x_norm))]
    y_correct = [max(list(zip(range(Y_LEN), y)), key=lambda x:x[1])[0]\
                 for y in y_test]

    # print(y_correct)
    # print(y_real_test)

    # print(sum([(1 if y[0] == y[1] else 0) for y in zip(y_predict, y_correct)]) / len(y_predict))

    print("tested findings:")
    print(sum([((y_to_num(y[0]) - 1) if y_to_num(y[0]) <= y[1] else -1) for y in zip(y_predict, y_real)]) / len(y_predict))
    print("testing EV max:")
    testPredict(y_real, 1, list(map(y_to_num, y_predict)), testing_model)
    print("testing most prob choice:")
    testPredict(y_real, 1, list(map(y_to_num, y_predict2)), "most_prob_cho")


    print("compared to:")
    # thresholds = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3, 3.5, 5, 7, 10, 20, 30, 50, 100, 150]
    thresholds = [1.1, 1.5, 2.2, 3.5, 10, 45]
    for thres in thresholds:
        testThreshold(y_real, 1, thres)

    


        
if __name__ == "__main__s":
    testModel()


if __name__ == "__main__":
    x_all = []
    x_raw_all = []
    y_all = []
    y_real = []
    check = False
    with open(f'behind{NUM_BEH}.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:

            added_row = list(float(x) for x in row[:NUM_BEH])
            added_row.sort()
            
            x_raw_all.append(added_row)
            # if not check:
            #     print(row)
            #     check = True
            #     print(x_all[-1])
            y_real.append(float(row[-1]))
            y_all.append(transform_num_y(float(row[-1])))
    


    
    for x in x_raw_all:
        x_all.append(list(map(transform_num, x)))
    

    # normalize the data
    mins = [0.0] * NUM_BEH
    maxes = [0.0] * NUM_BEH
    for i in range(0, NUM_BEH):
        mins[i] = min([x[i] for x in x_all])
        maxes[i] = max([x[i] for x in x_all])

    x_norm = [[(x[i] - mins[i]) / (maxes[i] - mins[i]) * 1.0 + 0.0 for i in range(0, NUM_BEH)] for x in x_all]

    # split into training data and test data
    test_size = int(len(x_norm) / 5)
    train_size = len(x_norm) - test_size

    x_train = np.matrix(x_norm[:train_size])
    y_train = np.matrix(y_all[:train_size])

    x_test = x_norm[train_size:]
    y_test = y_all[train_size:]
    y_real_test = y_real[train_size:]

    # set the topology of the neural network
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(12, activation="relu", input_dim = x_train.shape[1]))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation = "softmax"))

    # set up optimizer
    sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.15, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd)

    # train
    model.fit(x_train, y_train, epochs=15000, batch_size=100, verbose=1)

    model.save(testing_model)
    # #model: 3000, 100 0.228 profit
    # #model1: 5000, 200 0.4025 profit
    # #model2: 15000, 150 0.2105 profit
    # #model3: 5000, 150, 0.95 return

    # # beh2_model3 5000, 150, 0.95 return

    # model = tf.keras.models.load_model(testing_model)

            

    # get predictions, convert to class 0, 1, 2, and compare to test data
    y_predict = [getMax(y)[0]\
                 for y in model.predict(np.matrix(x_test))]
    y_predict2 = [max(list(zip(range(Y_LEN), y)), key=lambda x:x[1])[0]\
                 for y in model.predict(np.matrix(x_test))]
    y_correct = [max(list(zip(range(Y_LEN), y)), key=lambda x:x[1])[0]\
                 for y in y_test]

    # print(y_correct)
    # print(y_real_test)

    # print(sum([(1 if y[0] == y[1] else 0) for y in zip(y_predict, y_correct)]) / len(y_predict))

    print("tested findings:")
    print(sum([((y_to_num(y[0]) - 1) if y_to_num(y[0]) <= y[1] else -1) for y in zip(y_predict, y_real_test)]) / len(y_predict))
    print("testing EV max:")
    testPredict(y_real_test, 1, list(map(y_to_num, y_predict)), testing_model)
    print("testing most prob choice:")
    testPredict(y_real_test, 1, list(map(y_to_num, y_predict2)), "most_prob_cho")

    print("compared to:")
    thresholds = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3, 3.5, 5, 7, 10, 20, 30, 50, 100, 150]
    for thres in thresholds:
        testThreshold(y_real_test, 1, thres)

