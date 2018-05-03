import numpy as np

rewards = []
f = open("seed-rewards.out", "r")
lines = f.readlines()
for i in range(len(lines)):
    line_split = lines[i].split(" ")
    reward = float(line_split[4])
    rewards.append(reward)

rewards = np.array(rewards)
rewards.sort()
rewards = rewards[::-1]

print(rewards[:10])
