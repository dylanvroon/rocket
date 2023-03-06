import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel(3)
from analyze import testThreshold, testPredict

import numpy as np
import math
import csv
import random
#x_dividers log (Overall/PlayBatch)
#log5_1 = x_dividers = [1.1,1.24,1.47, 1.83, 2.44, 3.5, 5.4, 10.5], [1.1,1.5, 2, 3.5], sorted, lognormed, epoch = 5000
    # (1.08/0.87)
#log4_1 = same as log5_1
    # (1.03/0.95)
#log6_1
    #(1.08/0.89)
#log3_1
    #(1.06/0.92)
#log5_2 = same parameters as before, pessimisticly trained on PlayBatch
    #(0.99/1.21)
#log4_2
    #(1.01/1.18)
#log6_2
    #(1.01/1.18)
#5_3 = same parameters, trained on random 2/3 of total data set
    #(1.07/0.93)[1.01] No Zero: (1.07/0.94)
#6_3
    #(1.03/0.90)[0.95 (with a 1.05 mpc?)] No Zero: (1.06/0.94)
#4_3
    #(1.08/0.95) [1.11] No Zero: (1.09/1.00)

# x_dividers = [1.1,1.2,1.5, 2, 2.6, 3.5, 5, 10]
x_dividers = [1.1,1.24,1.47, 1.83, 2.44, 3.5, 5.4, 10.5]
X_LEN = len(x_dividers) + 1
x_code = 3


y_dividers = [1.1,1.5, 2, 3.5]
# y_dividers = [1.2,1.5,2,3.5]
# y_dividers = [1.1, 2, 3]
Y_LEN = len(y_dividers) + 1

NUM_BEH = 6

testing_model = f"behind{NUM_BEH}_logmodel{x_code}"
TRAINING_SET = f"behind{NUM_BEH}.csv"
TESTING_SET = f"behind{NUM_BEH}Play2.csv"
SET_DIVIDE = 3

global zero_count
zero_count = 0

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

def logRegA(x): #[1.1,1.24,1.45, 1.85, 2.5, 3.5, 5.3, 10]
        a = -1.046661
        b = 1.91722
        c = 0.867079
        d = -0.20295
        g = 0.185033
        return a*math.pow(math.log(g*x+c),d) + b


def getMax(probs, zeroDisabled=False): #list is of pairs of y_divider indicator and probability that one will occur
    sumProbs = list(map(lambda i:sum(probs[i:]), range(Y_LEN)))
    def getEV(pair):
        evWin = pair[1] * (y_to_num(pair[0]) - 1)
        evLoss = (1 - pair[1]) * -1
        return evWin + evLoss
    if random.random() < 0:
        print("Probs:")
        for prob in probs:
            print(f"\t{prob:,.3f}", end ="")
        print("\nSumProbs:")
        for prob in sumProbs:
            print(f"\t{prob:,.3f}", end ="")
        print("\nEvs:")
        for ev in list(map(getEV, list(zip(range(Y_LEN), sumProbs)))):
            print(f"\t{ev:,.3f}", end ="")
        print()
    myMax = max(list(zip(range(Y_LEN), sumProbs)), key=getEV)
    if zeroDisabled:
        if myMax[0] == 0:
            global zero_count
            print(f"Got 0: {zero_count}")
            zero_count += 1
        return max(list(zip(range(Y_LEN), sumProbs))[1:], key= getEV)
    else:
        return myMax

def testModel():
    x_norm = []
    y_test = []
    y_real = []
    with open(TESTING_SET, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            added_list = list(logRegA(float(x)) for x in row[:NUM_BEH])
            added_list.sort()
            x_norm.append(added_list)
            y_real.append(float(row[-1]))
            y_test.append(transform_num_y(float(row[-1])))

    # with open(f'behind{NUM_BEH}Play2.csv', newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for row in reader:
    #         added_list = list(logRegA(float(x)) for x in row[:NUM_BEH])
    #         added_list.sort()
    #         x_test.append(added_list)
    #         y_real.append(float(row[-1]))
    #         y_test.append(transform_num_y(float(row[-1])))


    model = tf.keras.models.load_model(testing_model)
    y_predict = [getMax(y, True)[0]\
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
    x_norm = []
    y_all = []
    y_real = []
    check = False
    with open(TRAINING_SET, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:

            added_row = list(logRegA(float(x)) for x in row[:NUM_BEH])
            added_row.sort()
            
            x_norm.append(added_row)
            # if not check:
            #     print(row)
            #     check = True
            #     print(x_all[-1])
            y_real.append(float(row[-1]))
            y_all.append(transform_num_y(float(row[-1])))
    


    # split into training data and test data
    test_size = int(len(x_norm) / SET_DIVIDE)
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
    model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=0)

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

