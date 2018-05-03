# NOTE: run test_seeds.py script before running this 

import numpy as np
from operator import itemgetter

rewards = []

# read output file from test_seeds.py
f = open("seed_rewards.out", "r")
lines = f.readlines()

# parse input string data
for i in range(len(lines)):
    line_split = lines[i].split(" ")
    reward = float(line_split[3])
    seed = line_split[7].rstrip()
    rewards.append((reward, seed,))

# sort array in descending order of rewards
rewards.sort(key=itemgetter(0))
rewards = rewards[::-1]

# print top 5 rewards with respective seeds
for reward in rewards[:10]:
    print("total reward = ", reward[0], " - seed = ", reward[1])
