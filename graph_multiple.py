import matplotlib.pyplot as plt
import numpy as np
import os



DIR = 'results'
MODEL = 'pendulum'

N_ITER = 1000

lines = []
for file in os.listdir(os.path.join(DIR, MODEL)):
    data = {}
    data['x'] = []
    data['y'] = []
    data['y_mean'] = []
    with open(os.path.join(DIR, MODEL, file)) as file:
        for idx, line in enumerate(file):
            line = line.strip().split(',')
            x = int(line[0])
            y = float(line[1])
            y_mean = float(line[2])
            data['x'].append(x)
            data['y'].append(y)
            data['y_mean'].append(y_mean)
            if idx == N_ITER:
                break
    lines.append(data)

print(lines)

rewards = [line['y_mean'] for line in lines]

# print(rewards)
min_values = np.min(rewards, axis=0)
max_values = np.max(rewards, axis=0)

mean_values = np.mean(rewards, axis=0)

episodes = range(len(lines[0]['y_mean']))

# Plotting
plt.figure(figsize=(10, 6))

# Plot filled regions for each model
plt.plot(episodes, mean_values, color='blue', label='Mean reward')


plt.fill_between(episodes, min_values, max_values, alpha=0.3, color='blue')


# Customize the plot
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title(f'Learning process for {MODEL}')
plt.legend(loc='lower right')
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Display the plot
plt.show()
