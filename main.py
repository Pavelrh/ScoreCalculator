import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('score.csv')

# loss function is just the error function
def lossFunction(m, b, points):
    totalError = 0
    for i in range(len(points)):
        x = points.iloc[i].X
        y = points.iloc[i].y
        # ** 2 --> square
        totalError += y - ((m * x) + b) ** 2
        totalError = totalError / len(points)


