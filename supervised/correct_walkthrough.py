from parse_walkthrough import Walkthrough
from jericho import *


# get instructions
#wt = Walkthrough("../walkthroughs/zork_sentence_walkthrough")
#instructions = wt.get_all_actions()

# play w the instructions
rom = "../z-machine-games-master/jericho-game-suite/curses.z5"
bindings = load_bindings(rom)
seed = bindings["seed"]
env = FrotzEnv(rom, seed=seed)

instruction_list = []
for action in bindings["walkthrough"].split("/"):
	instruction_list.append(action)

for instruction in instruction_list:
	print("\n\n\n>" + instruction)
	observation, reward, done, info = env.step(instruction)
	print(observation)

print(reward, len(instruction_list))



