import csv
from decimal import Decimal, InvalidOperation

def makeDataSet():
    nums = []
    while True:
        num = input()
        try:
            amt = Decimal(num)
        except InvalidOperation:
            break
        else:
            nums.append(amt)
    with open('test.csv', 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i, num in enumerate(nums):
            if i < len(nums) - 2:
                reswriter.writerow([float(num), float(nums[i+1]), float(nums[i+2])])
    with open('rawData.csv', 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for num in nums:
            reswriter.writerow([float(num)])

if __name__ == "__main__":
    makeDataSet()

#1.26 into 1.38 marks a new set
#1 should proceed 1.7 (before 1.74)