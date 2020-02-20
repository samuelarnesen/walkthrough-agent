from jericho import *
from torch.utils.data import Dataset
import copy, re
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

	def __init__(self, filename=None, rom_path=None):
		
		self.locations = []
		self.inventories = []
		self.observations = []
		self.actions = []
		self.descriptions = []
		self.wt = None
		self.section_num = 0 # for iterator
		self.internal_num = -1 # for iterator
		self.start = True # for iterator

		if filename != None and rom_path != None:
			self.load_from_walkthrough(filename, rom_path)

	def load_from_walkthrough(self, wt_filename, rom_path):

		self.wt = Walkthrough(wt_filename)
		sections = self.wt.get_sections()
		bindings = load_bindings(rom_path)
		seed = bindings["seed"]
		env = FrotzEnv(rom_path, seed=seed)

		def add(observation, items, location, action):
			self.observations.append(observation.rstrip(" \n"))
			item_list = []
			for item in items:
				item_list.append(item.name)
			self.inventories.append(item_list)
			self.locations.append(location)
			self.actions.append(action)

			inventories_str = ""
			if len(self.inventories) > 0:
				inventories_str = ", ".join(self.inventories[-1])

			total_description = self.observations[-1] + " | " + inventories_str + \
					" | " + self.locations[-1] + " | " + self.actions[-1]

			if len(self.descriptions[-1]) == len(sections[len(self.descriptions) - 1]["List"]):
				self.descriptions.append([])

			self.descriptions[-1].append(total_description)


		self.descriptions.append([])
		observation, _ = env.reset()
		items = env.get_inventory()
		location = env.get_player_location()
		add(observation, items, location.name, "")

		location_descriptions = {}

		for section in self.wt.get_sections():
			for action in section["List"]:
				observation, _, _, _ = env.step(action)
				items = env.get_inventory()
				location = env.get_player_location()

				if location.name not in location_descriptions and location.name in observation:
					generic_description = "\n".join(observation.split("\n\n")[0:2])
					location_descriptions[location.name] = generic_description[generic_description.index(location.name):]

				if observation.rstrip(" \n") == location.name:
					observation = location_descriptions[location.name]

				add(observation, items, location.name, action)

		self.descriptions.pop()
		env.close()

	def get_state_descriptions(self):
		return self.descriptions

	def get_instructions(self):
		return self.wt.get_sections()

	def __iter__(self):
		# potentially add resets here
		self.section_num = 0 
		self.internal_num = -1 
		self.start = True
		return self

	def __next__(self):
		section = self.wt.get_section(self.section_num)
		self.internal_num = (self.internal_num + 1) % len(section["List"])

		if self.internal_num == 0 and not self.start:
			self.section_num += 1
			if self.section_num >= self.wt.get_number_of_sections():
				raise StopIteration
			section = self.wt.get_section(self.section_num)

		instruction = section["Text"]
		state = self.descriptions[self.section_num][self.internal_num]
		action = section["List"][self.internal_num]
		self.start = False

		return instruction, state, action, self.internal_num == 0

	def section_generator(self):

		sections = self.wt.get_sections()
		longest_section_length = -1
		for i in range(len(sections)):
			longest_section_length = max(longest_section_length, self.wt.get_length_of_section(i))

		for i in range(longest_section_length):
			pairs = []
			for j, section in enumerate(sections):
				if i < self.wt.get_length_of_section(j):
					pairs.append([section["Text"], self.descriptions[j][i], section["List"][i], i==0])
				else:
					pairs.append(None)

			yield pairs



class Walkthrough_Dataset(Dataset):

	def __init__(self, wt_filenames=None, rom_paths=None, spm_path=None):
		self.states = []
		self.instructions = []
		self.actions = []
		self.sp = None
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.walkthroughs = []

		if spm_path != None:
			self.sp = spm.SentencePieceProcessor()
			self.sp.Load(spm_path)

		if wt_filenames != None and rom_paths != None:
			if wt_filenames == type(""):
				wt_filenames = [wt_filenames]
			if rom_paths == type(""):
				rom_paths = [rom_paths]

			for wt_filename, rom_path in zip(wt_filenames, rom_paths):

				super_wt = SuperWalkthrough(wt_filename, rom_path)
				self.walkthroughs.append(super_wt)
				for state_section, instruction_section in zip(super_wt.get_state_descriptions(), super_wt.get_instructions()):
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



