from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

rewards = []
with open('rewards.txt') as f:
    rewards = [float(x) for x in f.read().split(',')]

plt.plot(rewards)

plt.xlabel('Epoch')
plt.ylabel('Average reward')
plt.title('Average rewards over epochs')
plt.grid()
# plt.lege/nd()
plt.tight_layout()
plt.savefig('plots/rewards-long.png')