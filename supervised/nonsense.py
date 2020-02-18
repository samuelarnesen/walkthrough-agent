"""

import re


def check_matches(template, command):
	replaced_template = template.replace("OBJ", "(\w+(?:\s?\w+){0,2})")
	match_obj = re.fullmatch(replaced_template, command)
	return match_obj != None

templates = ["throw OBJ in OBJ", "land", "tell OBJ TMP", "eat OBJ"]
#command = "tell man \"throw red cake in pool of tears\""
command = "tell man to throw cake in pool of tears"

#quote_regex = re.compile("(\"[\w|\s]+\")")
quote_regex = re.compile("(?:(\"[\w|\s]+\"))|(?:to ([\w|\s]+))")

for template in templates:
	if "TMP" in template:
		quote_match = re.search(quote_regex, command)
		if quote_match != None:
			idx = 1 if quote_match.group(1) != None else 2
			inner_command = quote_match.group(idx).lstrip("\"").rstrip("\"")
			for inner_template in templates:
				if "TMP" not in inner_template:
					if check_matches(inner_template, inner_command):
						print(template.replace("TMP",  "\"" + inner_template + "\""))
	else:
		if check_matches(template, command):
			print(template)


"""




from parse_walkthrough import Walkthrough, SuperWalkthrough
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
#from utils import *

rom = "../z-machine-games-master/jericho-game-suite/zork2.z5"
binding = jericho.load_bindings(rom)
seed = binding["seed"]
env = FrotzEnv(rom, seed=seed)
#all_templates = TemplateActionGenerator(binding).templates
wt = Walkthrough(filename="../walkthroughs/zork2_super_walkthrough")
for instruction in wt.get_all_actions():
	observation, _, _, _ = env.step(instruction)
print(observation)
