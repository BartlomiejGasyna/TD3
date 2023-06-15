import numpy as np
import matplotlib.pyplot as plt

# Read the input file
data = []
with open('results/pendulum3_results.txt', 'r') as file:
    for line in file:
        line = line.strip().split(',')
        x = int(line[0])
        y = float(line[1])
        data.append((x, y))

# Extract x and y values
x_values = [item[0] for item in data]
y_values = [item[1] for item in data]

# Calculate the moving average
window_size = 100
y_values_ma = np.convolve(y_values, np.ones(window_size) / window_size, mode='valid')
x_values_ma = x_values[window_size//2 : -window_size//2 + 1]

# Plot the graph and moving average
# plt.plot(x_values, y_values, label='Original')
plt.plot(x_values_ma, y_values_ma, label='Moving Average (Window Size = 100)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Graph with Moving Average')
plt.legend()
plt.grid(True)

# Show the graph
plt.show()
