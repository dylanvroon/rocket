import csv

import random

from supervised_nn import NUM_BEH

CSV_NAME = f"behind{NUM_BEH}.csv"

def randomizeCSV():
    nums = []
    with open(CSV_NAME, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            nums.append(row)
    
    random.shuffle(nums)

    with open(CSV_NAME, 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in nums:
            reswriter.writerow(row)

if __name__ == "__main__":
    randomizeCSV()
        