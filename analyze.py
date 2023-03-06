import csv
from decimal import Decimal
from random import random

import os
from numpy import divide
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel(3)


csv_names = ["rawData1.csv", "rawData2.csv", "rawData3.csv", "rawData4.csv", "rawData5.csv", "rawData6.csv", "rawData7.csv", "testData.csv", "playData2.csv"]
# csv_names = ["rawData4.csv", "rawData5.csv", "rawData6.csv"]
# csv_names = ["testData.csv"]
# csv_names = ["playData2.csv"]



def getAllNums():
    nums = []
    count = 1
    while True:
        csv_name = f"rawData{count}.csv"
        try:
            nums += getNums(csv_name)
        except FileNotFoundError:
            break
        else:
            count += 1
    return nums

def getNums(csv_name = "rawData.csv"):
    nums = []
    #csv_name = input("Input csv file name:")
    try:
        csvfile = open(csv_name, newline='')
    except FileNotFoundError:
        raise FileNotFoundError
    else:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            try:
                nums.append(float(row[0]))
            except IndexError:
                pass
        return nums

def getAverage(nums):
    return sum(nums)/len(nums)

def testThreshold(nums, wager, thres):
    wins = 0
    losses = 0
    net = 0
    for num in nums:
        if num > thres:
            wins += (thres-1)*wager
        else:
            losses += wager
    net = wins - losses
    print(f"Threshold: {thres:02,.2f}", end="   \t | ")
    print(f"Net: {wins-losses:02,.2f}", end="\t | ")
    print(f"Total Return: {1 + net/(wager*len(nums)):02,.2f}")

def testPredict(nums, wager, predicted, model_name = ""):
    wins = 0
    losses = 0
    net = 0
    for i, num in enumerate(nums):
        if num > predicted[i]:
            wins += (predicted[i] - 1) * wager
        else:
            losses += wager
    net = wins - losses
    print(model_name, end = "       \t | ")
    print(f"Net: {wins-losses:02,.2f}", end="\t | ")
    print(f"Total Return: {1 + net/(wager*len(nums)):02,.2f}")
        

def countThreshold(nums, thres):
    less = 0
    great = 0
    for num in nums:
        if num < thres:
            less += 1
        else:
            great += 1
    print(f"Threshold: {thres:02,.2f}", end="   \t | ")
    print(f"Greater: {great:02,.2f}", end = "\t | ")
    print(f"Lesser: {less:02,.2f}")

def makeXbehind(numPrec):
    with open(f'behind{numPrec}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for csv_name in csv_names:
            nums = getNums(csv_name)
            for i in range(len(nums)):
                if i < len(nums) - numPrec:
                    row = map(lambda j: nums[i+j], range(numPrec + 1))
                    writer.writerow(row)
            nums.clear()

def countDividers(nums,dividers):
    names = {}
    for i,x in enumerate(dividers):
        if i == 0:
            names[i] = f"{1:,.2f}-{dividers[0]:,.2f}"
        else:
            names[i] = f"{dividers[i-1]:,.2f}-{dividers[i]:,.2f}"
    names[len(dividers)] = f"{dividers[-1]:,.2f}-inf"

    def transform_num(num):
        for i, x in enumerate(dividers):
            if num < x:
                return i
        return len(dividers)

    count = {}
    for i in range(len(dividers) + 1):
        count[i] = 0
    for num in nums:
        count[transform_num(num)] += 1

    for i in range(len(dividers) + 1):
        print(f"{names[i]}: \t {count[i]}")


def analyzeThresholds(thresholds, data):
    print("Return rates ____________________")
    for thres in thresholds:
        testThreshold(data, 1, thres)
    # print()
    # print("Counting ______________________")
    # for thres in thresholds:
    #     countThreshold(data, thres)

if __name__ == "__main__":
    makeXbehind(2)
    makeXbehind(3)
    makeXbehind(4)
    makeXbehind(5)
    makeXbehind(6)
    data = getAllNums()
    print(len(data))
    print(f"Average: {getAverage(data):,.4f}")
    countDividers(data, [1.1,1.24,1.47, 1.83, 2.44, 3.5, 5.4, 10.5])
    # countDividers(data, [1.2,1.5, 2, 3.5])

    # thresholds = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.5, 7, 8, 9, 10, 20, 30, 50, 100, 150]
    
    # predicted = []
    # # thresholds = [1.1,3.5]
    # # for x in range(15):
    # #     num = random() * random() * random() * random() * random() * 1000 + 1
    # #     thresholds.append(num)
    
    # # thresholds.sort()
    # analyzeThresholds(thresholds, data)
    # for y in range(3):
    #     for x in range(696):
    #         num = random() * random() * random() * random() * random() * 100 + 1
    #         predicted.append(num)
    #     testPredict(data, 1, predicted, "fucking_around")
    #     predicted.clear()
    