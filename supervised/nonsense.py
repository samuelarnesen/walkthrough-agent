from parse_walkthrough import SuperWalkthrough
from jericho import *

rom_list = ["../z-machine-games-master/jericho-game-suite/zork1.z5", "../z-machine-games-master/jericho-game-suite/zork2.z5", "../z-machine-games-master/jericho-game-suite/zork3.z5"]
wt_list = ["../walkthroughs/zork_super_walkthrough", "../walkthroughs/zork2_super_walkthrough", "../walkthroughs/zork3_super_walkthrough"]


swt = SuperWalkthrough(wt_list, rom_list)

for section in swt.section_generator():
	pass



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