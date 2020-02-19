import re


action = "tell dungeon master to go to parapet"
template = "go to parapet"
upper_template = "tell OBJ to CMD"

ut2 = upper_template.replace("OBJ", "(\w+(?:\s?\w+){0,3}?)").replace("CMD", "(.*)")


print(ut2)
match_obj = re.search(re.compile(ut2), action)

if match_obj != None:
	print(match_obj.groups())
else:
	print("No Match")



"""
from parse_walkthrough import Walkthrough, SuperWalkthrough
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
#from utils import *

rom = "../z-machine-games-master/jericho-game-suite/zork3.z5"
binding = jericho.load_bindings(rom)
seed = binding["seed"]
env = FrotzEnv(rom, seed=seed)
#all_templates = TemplateActionGenerator(binding).templates
wt = Walkthrough(filename="../walkthroughs/zork3_super_walkthrough")
for instruction in wt.get_all_actions():
	observation, _, _, _ = env.step(instruction)
print(observation)
"""