import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

results = pd.read_csv("progress.csv", header=None).values
rows, cols = np.shape(results)

loss = []
for i in range(5, rows):
	loss.append(max(float(results[i, -1]), -10))

period = 15
current = 0
for i in range(0, period):
	current += loss[i]

moving_average_values = [current / period]
for i in range(period, len(loss)):
	current += (loss[i] - loss[i - period])
	moving_average_values.append(current / period)


plt.plot(range(period, len(moving_average_values) + period), moving_average_values, c="r")
plt.scatter(range(len(loss)), loss, s=5)
plt.show()