from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
from torch.utils.data import Dataset
import copy, re, sys
import sentencepiece as spm
import numpy as np
import torch 
from utils import *

class Walkthrough:

	def __init__(self, filename=None):
		self.sections = []
		self.all_instructions=[]
		if filename != None:
			self.load_from_file(filename)

	def load_from_file(self, filename):

		self.sections = []

		current_section = {"List": []}
		with open(filename) as f:
			for line in f.readlines():

				# skip empty lines
				if len(line) < 2:
					continue 

				# skip the citation lines
				if len(line) > 2:
					if line[0] == "#" and line[-1] == "#":
						continue

				if line[0] not in ["?", ">"]:
					if len(current_section["List"]) > 0:
						self.sections.append(current_section)
						current_section = {"List": []}
					current_section["Text"] = line.rstrip(" \n")
				else:
					current_section["List"].append(line.lstrip(" ?>").rstrip(" \n").replace("(?)", ""))

		if len(current_section["List"]) > 0:
			self.sections.append(current_section)

		for section in self.sections:
			for action in section["List"]:
				self.all_instructions.append(action)

	def get_sections(self):
		return self.sections

	def get_number_of_sections(self):
		return len(self.sections)

	def get_all_actions(self):
		return self.all_instructions

	def get_number_of_actions(self):
		return len(self.all_instructions)

	def get_section(self, idx):
		if idx >= len(self.sections) or idx < 0:
			return None
		return self.sections[idx]

	def get_length_of_section(self, idx):
		if idx >= len(self.sections) or idx < 0:
			return None
		return len(self.sections[idx]["List"])

	def print_section(self, idx):
		section = self.get_section(idx)
		if section == None:
			print("Not a section -- sections must be in range 0-" + str(len(self.sections)))
			return
		print(section["Text"])
		print()
		for instruction in section["List"]:
			print(instruction)

class SuperWalkthrough:

	def __init__(self, filename=None, rom_path=None, requires_valid_actions=False):
		
		self.initial_observations = []
		self.inventories = []
		self.observations = []
		self.actions = []
		self.descriptions = []
		self.wt = []

		self.section_num = 0 # for iterator
		self.internal_num = -1 # for iterator
		self.start = True # for iterator
		self.requires_valid_actions = requires_valid_actions
		self.number_of_sections = 0

		self.states_in_order = []
		self.valid_actions = []

		if filename != None and rom_path != None:
			self.load_from_walkthrough(filename, rom_path)

	def load_from_walkthrough(self, wt_filenames, rom_paths):

		def add(observation, items, initial_observation, action, start):
			self.observations.append(observation.rstrip(" \n"))
			item_list = []
			for item in items:
				item_list.append(item.name)
			self.inventories.append(item_list)
			self.initial_observations.append(initial_observation)
			self.actions.append(action)

			inventories_str = ""
			if len(self.inventories) > 0:
				inventories_str = ", ".join(self.inventories[-1])

			total_description = self.observations[-1] + " | " + inventories_str + \
					" | " + self.initial_observations[-1] + " | " + self.actions[-1]

			if start:
				self.descriptions.append([])
				self.number_of_sections += 1

			self.descriptions[-1].append(total_description)


		if type(wt_filenames) != type([]):
			wt_filenames = [wt_filenames]
		if type(rom_paths) != type([]):
			rom_paths = [rom_paths]

		for wt_filename, rom_path in zip(wt_filenames, rom_paths):

			tag = TemplateActionGenerator(jericho.load_bindings(rom_path))

			self.wt.append(Walkthrough(wt_filename))
			sections = self.wt[-1].get_sections()
			bindings = load_bindings(rom_path)
			seed = bindings["seed"]
			env = FrotzEnv(rom_path, seed=seed)

			observation, _ = env.reset()
			items = env.get_inventory()
			location = env.get_player_location()
			add(observation, items, location.name, "", True) # fix

			counter = 0
			for i, section in enumerate(sections):
				for j, action in enumerate(section["List"]):

					initial_observation, _, _, _ = env.step(action)

					items = env.get_inventory()
					observation, _, _, _ = env.step("look")

					if self.requires_valid_actions:
						if len(self.valid_actions) > 0:
							if action not in self.valid_actions[-1]:
								self.valid_actions[-1].append(action)
							print(counter, "\t", self.valid_actions[-1])

						# gets all the objects
						objs = env.identify_interactive_objects(initial_observation)
						for obj in env.identify_interactive_objects(observation):
							if obj not in objs:
								objs.append(obj)
						obj_list = [obj[0] for obj in objs]
						possible_actions = tag.generate_actions(obj_list)
						self.valid_actions.append(env.find_valid_actions(possible_actions))

					env.reset()
					for i2 in range(0, i + 1):
						section2 = sections[i2]
						for j2, action2 in enumerate(section2["List"]):
							env.step(action2)
							if i2 == i and j2 == j:
								break

					add(observation, items, initial_observation, action, j==len(section["List"])-1)
					counter += 1
					
			self.descriptions.pop()
			env.close()

		for block in self.descriptions:
			for ob in block:
				self.states_in_order.append(ob.replace("\n", " "))

	def get_state_descriptions(self):
		return self.states_in_order

	def get_state_descriptions_by_section(self):
		return self.descriptions

	def get_instructions(self):
		sections = []
		for ind_wt in self.wt:
			for section in ind_wt.get_sections():
				sections.append(section)
		return sections		

	def get_actions(self):
		return self.actions[1:]

	def get_valid_actions(self):
		return self.valid_actions

	def __len__(self):
		sections = self.get_instructions()
		return len(sections)

	def __iter__(self):
		self.section_num = 0 
		self.internal_num = -1 
		self.start = True
		self.number_of_sections = len(self.descriptions)
		self.wt_num = 0

		return self

	def __next__(self):

		internal_section_num = self.section_num
		for i in range(0, self.wt_num):
			internal_section_num -= self.wt[i].get_number_of_sections()

		wt = self.wt[self.wt_num]
		section = wt.get_section(internal_section_num)
		self.internal_num = (self.internal_num + 1) % len(section["List"])

		if self.internal_num == 0 and not self.start:
			self.section_num += 1
			internal_section_num += 1
			if internal_section_num >= wt.get_number_of_sections():
				self.wt_num += 1
				internal_section_num = 0
				if self.wt_num >= len(self.wt):
					raise StopIteration
			section = wt.get_section(internal_section_num)


		instruction = section["Text"]
		state = self.descriptions[self.section_num][self.internal_num]
		action = section["List"][self.internal_num]
		self.start = False

		return instruction, state, action, self.internal_num == 0

	def section_generator(self):

		longest_section_length = -1
		section_lengths = []
		for wt in self.wt:
			sections = wt.get_sections()
			for i in range(len(sections)):
				longest_section_length = max(longest_section_length, wt.get_length_of_section(i))
				section_lengths.append(wt.get_length_of_section(i))

		for i in range(longest_section_length):
			pairs = []
			section_base = 0
			for wt in self.wt:
				sections = wt.get_sections()
				for j, section in enumerate(sections):
					if i < section_lengths[section_base + j]:
						pairs.append([section["Text"], self.descriptions[section_base + j][i], section["List"][i], i==0])
					else:
						pairs.append(None)
				section_base += len(sections)

			yield pairs

class Walkthrough_Dataset(Dataset):

	def __init__(self, wt_filenames=None, rom_paths=None, spm_path=None):
		self.states = []
		self.instructions = []
		self.actions = []
		self.sp = None
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.walkthroughs = []
		self.wt_idxs = []

		if spm_path != None:
			self.sp = spm.SentencePieceProcessor()
			self.sp.Load(spm_path)

		if wt_filenames != None and rom_paths != None:
			if wt_filenames == type(""):
				wt_filenames = [wt_filenames]
			if rom_paths == type(""):
				rom_paths = [rom_paths]

			for wt_filename, rom_path in zip(wt_filenames, rom_paths):
				self.wt_idxs.append(len(self.actions))

				super_wt = SuperWalkthrough(wt_filename, rom_path)
				self.walkthroughs.append(super_wt)
				for state_section, instruction_section in zip(super_wt.get_state_descriptions_by_section(), super_wt.get_instructions()):
					for state, action in zip(state_section, instruction_section["List"]):
						self.states.append(state)
						self.instructions.append(instruction_section["Text"])
						self.actions.append(action)

				if spm_path != None:
					game_states = convert_batch_to_tokens(self.states, 200, self.device, self.sp)
					game_instructions = convert_batch_to_tokens(self.instructions, 400, self.device, self.sp)
					if self.states == type([]):
						self.states = game_states
						self.instructions = game_instructions
					else:
						self.states = torch.cat([self.states, game_states], dim=0)
						self.instructions = torch.cat([self.instructions, game_instructions], dim=0)


	def __getitem__(self, index):
		return ((self.states[index], self.instructions[index]), self.actions[index])

	def __len__(self):
		return len(self.states)


	def split(self, indices):
		split_wtd = Walkthrough_Dataset()

		for idx in indices:
			split_wtd.states.append(self.states[idx])
			split_wtd.instructions.append(self.instructions[idx])
			split_wtd.actions.append(self.actions[idx])

		return split_wtd

	def split_off_final_wt(self):
		return self.split(range(self.wt_idxs[-1])), self.split(range(self.wt_idxs[-1], len(self)))


class LinkedWalkthrough:

	def __init__(self, swt, text_path):
		swt = copy.deepcopy(swt)
		wt = Walkthrough(text_path)

		self.total_list = []
		self.repeaters = []
		self.length = len(swt)

		swt_idx = 0
		swt_actions = swt.get_actions()
		swt_states = swt.get_state_descriptions()

		for i, section in enumerate(wt.get_sections()):
			current_instruction = section["Text"]
			for j, action in enumerate(section["List"]):

				act_to_use = action
				current_repeater = None
				num_repeats = 1
				if action != swt_actions[swt_idx]:
					repeat_match = re.match("repeat (.*) until (.*) \((\d+)\)", action)
					if repeat_match == None:
						print("BROKEN: ", action, "\t", swt_actions[swt_idx])
						sys.exit()
					assert(repeat_match != None)
					act_to_use = repeat_match.group(1)
					terminal_condition = repeat_match.group(2)
					num_repeats = int(repeat_match.group(3))
					current_repeater = Repeater(action=act_to_use, num_repeats=num_repeats, terminal_condition=terminal_condition)

				while num_repeats > 0:
					if current_repeater != None:
						current_repeater.states.append(swt_states[swt_idx])
						assert(len(current_repeater.states) <= current_repeater.num_repeats)
						#step_instruction, step_state, step_action, step_start
						if num_repeats == current_repeater.num_repeats:
							self.total_list.append([current_instruction, swt_states[swt_idx], act_to_use, j==0])
					else:
						self.total_list.append([current_instruction, swt_states[swt_idx], act_to_use, j==0])
					swt_idx += 1
					num_repeats -= 1

				if current_repeater != None:
					current_repeater.states.append(swt_states[swt_idx]) # add the first legit one
					self.repeaters.append(current_repeater)

	def __len__(self):
		return self.length

	def __iter__(self):
		self.idx = -1
		return self

	def __next__(self):
		if self.idx >= len(self.total_list) - 1:
			raise StopIteration
		self.idx += 1
		return self.total_list[self.idx]


# utility class for sequences that repeat
class Repeater:
	def __init__(self, action="", num_repeats=0, terminal_condition="", states=None):
		self.action = action
		self.num_repeats = num_repeats
		self.terminal_condition = terminal_condition
		self.states = states if states != None else []

	def __str__(self):
		return "Action: {}\nRepeats: {}\nTerminal Condition: {}\nStates:\n\n{}".format(self.action, \
			self.num_repeats, self.terminal_condition, "\n\n".join(self.states))

