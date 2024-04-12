import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('score.csv')

plt.scatter(data.X, data.Y)

plt.show()