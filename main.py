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

# this function calculates loss and minimises it unlike the lossFunction (refer to notes for mathematical explanation)
def gradientDescent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].X
        y = points.iloc[i].Y
        # refer to notes for mathematical explanation
        m_gradient += (-2/n) * x * (y - ((m_now * x) + b_now))
        b_gradient += (-2/n) * (y - ((m_now * x) + b_now))

    # L -> learning rate (size of steps towards minimising the loss function)
    m = m_now - L * m_gradient
    b = b_now - L * b_gradient    

    return m, b


m = 0
b = 0

# L -> learning rate (size of steps towards minimising the loss function)
L = 0.001
# epoch -> number of iterations
epoch = 300

# this will constantly get better and better at calculating the best m & b from each iteration
for i in range(epoch):
    # prints epoch value every 50 iterations
    if i % 50 == 0:
        print(f"Epoch (iteration) : {i}")
    # assigning the returned m and b to the actual m and b
    m,b = gradientDescent(m, b, data, L)

print(m, b)

plt.scatter(data.X, data.Y, color="black")
plt.plot(list(range(2, 10)), [m * x + b for x in range(2,10)], color="red")
plt.show()