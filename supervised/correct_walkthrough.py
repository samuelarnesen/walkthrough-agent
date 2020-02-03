from parse_walkthrough import Walkthrough
from jericho import *

# get instructions
instructions = []
with open("../walkthroughs/zork_ts_0") as f:
	for line in f.readlines():
		if len(line) < 2:
			continue
		if line[0] == ">":
			instructions.append(line.lstrip(" >").rstrip(" \n"))

# check the walkthrough is the same
"""wt = Walkthrough("../walkthroughs/zork_super_walkthrough")
super_instructions = wt.get_all_actions()
for action1, action2 in zip(super_instructions, instructions):
	print(action1)
	if action1 != action2:
		print("\nERROR:")
		print("super:", action1, "\ncorrect:", action2)
		break
"""

# play w the instructions
rom = "zork1.z5"
bindings = load_bindings(rom)
seed = bindings["seed"]
env = FrotzEnv(rom, seed=seed)

for act in instructions:
	observation, reward, done, info = env.step(act)
	print(">" + act)
	print(observation)





