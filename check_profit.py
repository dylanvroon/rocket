import csv

from analyze import testPredict, analyzeThresholds
from sup_nn_log import testing_model
CSV_NAME = "propData3.csv"

if __name__ == "__main__":
    proposed = []
    correct = []
    thresholds = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.5, 7, 8, 9, 10, 20, 30, 50, 100, 150]
    with open(CSV_NAME, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            proposed.append(float(row[0]))
            correct.append(float(row[1]))
    testPredict(correct, 1, proposed, testing_model)
    analyzeThresholds(thresholds, correct)