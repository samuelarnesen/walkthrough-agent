
weird_list = [124, 12414, 45345, None, 2313, 3523, None ,12145, 3463434, None, None, 23]

transformed = list(i for i, item in enumerate(weird_list) if item is not None)


print(transformed)


"""
import re

template_string = "strike OBJ"
action_to_use = "strike thief with knife"

lookarounds = ["with", "to", "from", "at", "in", "under"]
lookaheads = "".join("(?!\\b" + la + "\\b)" for la in lookarounds)
lookbehinds = "".join("(?<!\\b" + la + "\\b)" for la in lookarounds)

match_obj = re.fullmatch(template_string.replace("OBJ", lookaheads + "(\w+(?:\s?\w+){0,3}?)" + lookbehinds).replace("\w", "(?:" + lookaheads + "\w)"), action_to_use)

print(template_string.replace("OBJ", lookaheads + "(\w+(?:\s?\w+){0,3}?)" + lookbehinds))
if match_obj != None:
	print(match_obj.groups())
else:
	print("None")

"""


"""
from parse_walkthrough import Walkthrough, SuperWalkthrough
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
#from utils import *

rom = "../z-machine-games-master/jericho-game-suite/zork3.z5"
bindings = load_bindings(rom)
seed = bindings["seed"]
env = FrotzEnv(rom, seed=seed)
#all_templates = TemplateActionGenerator(binding).templates
wt = Walkthrough("../walkthroughs/zork3_super_walkthrough")

for action in wt.get_all_actions():
	ob, _, _, _ = env.step(action)
print(ob)
"""