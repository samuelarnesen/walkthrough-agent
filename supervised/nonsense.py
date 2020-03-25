from parse_walkthrough import SuperWalkthrough
from jericho import *
import sys

rom_list = ["../z-machine-games-master/jericho-game-suite/zork1.z5", "../z-machine-games-master/jericho-game-suite/zork2.z5", "../z-machine-games-master/jericho-game-suite/zork3.z5"]
wt_list = ["../walkthroughs/zork_super_walkthrough", "../walkthroughs/zork2_super_walkthrough", "../walkthroughs/zork3_super_walkthrough"]
swt = SuperWalkthrough(wt_list, rom_list)


def get_source_and_target(section):
	"""
	print(section["Text"].lower())
	print()
	print(section["List"])
	print()
	"""

	source = section["Text"].lower().split()
	target = []
	for action in section["List"]:
		for word in action.split():
			target.append(word)
		target.append("\n")
	target.pop()

	return source, target

def realize(source, tags):
	target = []
	source_idx = 0
	for i, tag in enumerate(tags):
		if type(tag) == type([]):
			for word in tag:
				if word == "KEEP":
					target.append(source[source_idx])
				else:
					target.append(word)
		elif tag == "KEEP":
			target.append(source[source_idx])
		source_idx += 1

	for i, word in enumerate(target):
		target[i] = word.strip(" ,.:)(")
	return target

def find_next_command(target, idx):
	for i in range(idx + 1, len(target)):
		if target[i] == "\n":
			return i + 1
	return len(target)

def find_next_match(source, source_idx, target, target_idx, max_phrase_length):
	p = []
	match_found = False
	for j in range(0, max_phrase_length):
		if target_idx + j >= len(target):
			break
		p.append(target[target_idx + j])
		if source[source_idx].lower().strip(" .)(,") == target[target_idx + j].lower().strip(" .)(,"):
			match_found = True
			p.pop()
			p.append("KEEP")
			break
	return p, match_found

def merge_tags(tags):
	first = True
	for i, tag in enumerate(tags):
		if type(tag) == type([]) and not first:
			if tag[0] != "\n":
				word = tag[0]
				tags[i] = tag[1:]
				idx = i - 1
				while idx >= 0:
					if type(tags[idx]) == type([]):
						tags[idx].append(word)
						break
					elif tags[idx] == "KEEP":
						tags[idx] = ["KEEP", word]
						break
					idx -= 1
			first = False
		elif tag == "KEEP":
			first = False
	return tags


MAX_PHRASE_LENGTH = 3
wt = swt.wt[0]
section = wt.get_section(1)

source, target = get_source_and_target(section)

tags = []
for i in range(len(source)):
	tags.append("DEL")
source_idx = 0
target_idx = 0


while target_idx < len(target):

	if source_idx >= len(source):
		break

	if source[source_idx].lower().strip(" .)(,") == target[target_idx].lower().strip(" .)(,"):
		tags[source_idx] = "KEEP"
		target_idx += 1
	else:
		p, match_found = find_next_match(source, source_idx, target, target_idx, MAX_PHRASE_LENGTH)
		if match_found:
			if len(p) > 0:
				tags[source_idx] = p
			target_idx += len(p)
		else:
			temp_target_idx = find_next_command(target, target_idx)
			if temp_target_idx < len(target):
				p, match_found = find_next_match(source, source_idx, target, temp_target_idx, MAX_PHRASE_LENGTH)
				if match_found:
					if len(p) > 0:
						previous = target[target_idx:temp_target_idx]

						for element in p:
							previous.append(element)
						tags[source_idx] = previous
					target_idx = temp_target_idx + len(p) - 1
	source_idx += 1

tags = merge_tags(tags)

for tag, word in zip(tags, source):
	print(word, "-", tag)


	
target_guess = realize(source, tags)
missing = []

guess_idx = 0
correct_idx = 0
while correct_idx < len(target):

	if guess_idx >= len(target_guess):
		for j in range(correct_idx, len(target)):
			missing.append(target[j])
		break

	if target_guess[guess_idx] == target[correct_idx]:
		guess_idx += 1
	else:
		missing.append(target[correct_idx])
	correct_idx +=1

"""
print()
print(target)
print()
print(target_guess)
print()
print(" ".join(missing).split("\n"))
"""





