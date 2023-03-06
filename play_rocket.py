
import csv
from supervised_nn import y_to_num
from sup_nn_log import getMax, logRegA
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel(3)

MODEL_NAME = "behind4_logmodel3"
NUM_BEH = 4

if __name__ == "__main__":
    nums = []
    model = tf.keras.models.load_model(MODEL_NAME)
    proposed = None
    with open('playData3.csv', 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        with open('propData3.csv', 'w', newline='') as csvfile2:
            propwriter = csv.writer(csvfile2, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            while True:
                num = input()
                try:
                    val = float(num)
                except ValueError:
                    break
                # reswriter.writerow([val])
                # if proposed:
                #     propwriter.writerow([proposed, val])
                nums.append(val)
                if len(nums) > NUM_BEH:
                    x_data = list(map(logRegA ,nums[len(nums) - NUM_BEH: len(nums)]))
                    # for i, x in enumerate(x_data):
                    #     print(f"\t{nums[len(nums) - NUM_BEH + i]}: {x:,.2f}", end = "")
                    # print()
                    x_data.sort()
                    probs = model.predict(np.matrix([x_data]), verbose=0)[0]
                    # for prob in probs:
                    #     print(f"\t{prob:,.4f}", end = "")
                    # print()
                    proposed = y_to_num(getMax(probs, True)[0])
                    print(f"\t{proposed}")
                    x_data.clear()




                
