from parse_walkthrough import Walkthrough_Dataset, SuperWalkthrough
from utils import *
from model import BasicModel, TimeModel
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler

import numpy as np
import sentencepiece as spm
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
import random, re, math, sys

class Agent_Zork:

	def __init__(self, args, model_type="basic", model_name=None, index_file_name=None, save_name="basic_model.pt"):

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.data = Walkthrough_Dataset(args["walkthrough_filename"], args["rom_path"])
		self.templates = create_templates_list(args["rom_path"], args["temp_path"])
		self.vocab, self.reverse_vocab = load_vocab(args["rom_path"], args["add_word_path"])
		self.batch_size = args["batch_size"]
		self.num_epochs = args["num_epochs"]
		self.rom_paths = args["rom_path"]
		self.walkthrough_filenames = args["walkthrough_filename"]
		self.clip = args["clip"]
		self.learning_rate = args["learning_rate"]
		self.save_name = save_name
		self.o1_start = args["o1_start"]
		self.o2_start = args["o2_start"]

		# only relevant for if model is of the basic type -- splits the data into training, validation, and testing
		if index_file_name == None:
			shuffled_idxs = list(range(len(self.data)))
			random.shuffle(shuffled_idxs)
			train_idx_end = int(0.75 * len(self.data))
			val_idx_end = int(0.875 * len(self.data))
			self.train_data = self.data.split(shuffled_idxs[:train_idx_end])
			self.val_data = self.data.split(shuffled_idxs[train_idx_end:val_idx_end])
			self.test_data = self.data.split(shuffled_idxs[val_idx_end:])
			dump_indices("data_indices", shuffled_idxs[:train_idx_end], shuffled_idxs[train_idx_end:val_idx_end], shuffled_idxs[val_idx_end:])
		else:
			train_idxs, val_idxs, test_idxs = load_indices(index_file_name)
			self.train_data = self.data.split(train_idxs)
			self.val_data = self.data.split(val_idxs)
			self.test_data = self.data.split(test_idxs)

		sp = spm.SentencePieceProcessor()
		sp.Load(args["spm_path"])

		self.model_args = {
			"embedding_size": args["embedding_size"],
			"hidden_size": args["hidden_size"],
			"template_size": len(self.templates),
			"spm_path": args["spm_path"],
			"vocab_size": len(sp),
			"output_vocab_size": len(self.vocab) + 2,
			"batch_size": args["batch_size"],
			"max_number_of_sentences": args["max_number_of_sentences"],
			"max_number_of_words": args["max_number_of_words"]
		}
		self.model = BasicModel(self.model_args).to(self.device) if model_type == "basic" else TimeModel(self.model_args).to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=args["learning_rate"])

		if model_name != None:
			self.model.load_state_dict(torch.load(model_name))

	def train(self):
		"""
		trains and validates the model
		"""
		if self.model.get_name() == "basic":
			self.train_basic()
		elif self.model.get_name() == "time":
			self.train_time()
		else:
			print("METHOD NOT AVAILABLE")

	def train_time(self, track_accuracy=False):
		"""
		training time model
		"""

		def get_pct(correct, guesses):
			"""
			checks how many of the guesses are correct
			"""
			correct_count = 0
			for i, correct_item in enumerate(correct):
				if guesses[i] == correct_item:
					correct_count += 1

			return correct_count

		def get_accuracy_of_batch(q_ts, q_o1s, q_o2s, correct_ts, correct_o1s, correct_o2s):
			"""
			gets the accuracy of the entire batch -- only calculates for the template at the moment
			"""
			t = torch.argmax(q_ts, dim=1)

			return get_pct(correct_ts, t)

		def get_object_index(obj_indices):
			"""
			gets the index in the vocabulary of the object in multi-word object (e.g. "Dungeon Master" -> "Master")
			"""
			full_word = " ".join(self.vocab[idx] for idx in obj_indices)
			keyword = extract_object(full_word)
			if len(keyword) == 0:
				return None
			return self.reverse_vocab[keyword]

		def get_indices(pair, sentence_atts=[], word_atts=[]):
			"""
			gets the sections from the pair
			"""
			states = []
			instructions = []
			template_idxs = []
			o1_idxs = []
			o2_idxs = []
			sections_used = []
			sections_missing = []
			for i, section in enumerate(pair):
				if section != None:
					sections_used.append(i)
					instructions.append(section[0])
					states.append(section[1])
					template_idx, o1_idx, o2_idx = self.identify_components(section[2])

					template_idxs.append(template_idx)
					o1_idxs.append(get_object_index(o1_idx))
					o2_idxs.append(get_object_index(o2_idx))

					if section[3]:
						sentence_atts.append(None)
						word_atts.append(None)
				else:
					sections_missing.append(i)
			return states, instructions, template_idxs, o1_idxs, o2_idxs, sections_used, sections_missing, sentence_atts, word_atts

		def get_loss(qs, idxs, criterion):
			"""
			gets loss given guesses and correct ones
			"""

			sections = list(i for i, item in enumerate(idxs) if item is not None)
			idxs_without_nones = list(item for item in idxs if item is not None)
			if len(sections) > 0:
				idxs_to_use = torch.tensor(sections, dtype=torch.long)
				qs_to_use = torch.index_select(qs, 0, idxs_to_use)

				return criterion(qs_to_use, torch.tensor(idxs_without_nones, dtype=torch.long)), qs_to_use
			return 0, None

		def execute_epoch(swt, t_criterion, o1_criterion, o2_criterion):
			"""
			makes predictions and calculates loss
			"""

			t_loss = []
			o1_loss = []
			o2_loss = []

			sentence_atts = []
			word_atts = []
			num_examples = {"t": 0, "o1": 0, "o2": 0}
			correct_count = 0
			for i, pair in enumerate(swt.section_generator()):

				if i > 30:
					break

				states, instructions, template_idxs, o1_idxs, o2_idxs, sections_used, sections_missing, sentence_atts, word_atts = get_indices(pair, sentence_atts, word_atts)

				sentence_atts = torch.index_select(sentence_atts, 0, torch.tensor(sections_used, dtype=torch.long)) if torch.is_tensor(sentence_atts) else sentence_atts
				word_atts = torch.index_select(word_atts, 0, torch.tensor(sections_used, dtype=torch.long)) if torch.is_tensor(word_atts) else word_atts
				q_ts, q_o1s, q_o2s, sentence_atts, word_atts = self.model(states, instructions, sentence_atts, word_atts)
				
				batch_t_loss, _ = get_loss(q_ts, template_idxs, t_criterion)
				batch_o1_loss, q_o1s_to_use = get_loss(q_o1s, o1_idxs, o1_criterion)
				batch_o2_loss, q_o2s_to_use = get_loss(q_o2s, o2_idxs, o2_criterion)

				t_loss.append(batch_t_loss)
				o1_loss.append(batch_o1_loss)
				o2_loss.append(batch_o2_loss)

				with torch.no_grad():
					correct_count += get_accuracy_of_batch(q_ts, q_o1s_to_use if len(o1_idxs) > 0 else None, q_o2s_to_use if len(o2_idxs) > 0 else None, template_idxs, o1_idxs, o2_idxs)
				num_examples["t"] += len(template_idxs)
				num_examples["o1"] += len(o1_idxs)
				num_examples["o2"] += len(o2_idxs)

				for i in sections_missing:
					sentence_atts = torch.cat([sentence_atts[0:i, :], torch.zeros([1, self.model_args["max_number_of_sentences"]]), sentence_atts[i:, :]])
					word_atts = torch.cat([word_atts[0:i, :, :], torch.zeros([1, self.model_args["max_number_of_sentences"], self.model_args["max_number_of_words"]]), word_atts[i:, :, :]])

			return sum(t_loss), sum(o1_loss), sum(o2_loss), num_examples, correct_count

		def backward(total_t_loss, total_o1_loss, total_o2_loss, num_examples):
			"""
			updates parameters
			"""
			total_t_loss.backward() 
			self.optimizer.step()
			self.optimizer.zero_grad()

		def validate(val_swt, t_criterion, o1_criterion, o2_criterion):
			""" 
			validate
			"""
			with torch.no_grad():
				total_t_loss, total_o1_loss, total_o2_loss, num_examples, correct_count = execute_epoch(val_swt, t_criterion, o1_criterion, o2_criterion)
				print("Val", str(0), "\t", total_t_loss.item() / num_examples["t"], "\t", correct_count / num_examples["t"])
				self.optimizer.zero_grad()

		## ACTUAL EXECUTION CODE

		t_criterion = nn.NLLLoss(reduction='sum')
		o1_criterion = nn.NLLLoss(reduction='sum')
		o2_criterion = nn.NLLLoss(reduction='sum')

		train_swt = SuperWalkthrough(self.walkthrough_filenames[0:-1], self.rom_paths[0:-1])
		val_swt = SuperWalkthrough([self.walkthrough_filenames[-1]], [self.rom_paths[-1]])


		validate(val_swt, t_criterion, o1_criterion, o2_criterion)

		for epoch in range(self.num_epochs):
			total_t_loss, total_o1_loss, total_o2_loss, num_examples, correct_count = execute_epoch(train_swt, t_criterion, o1_criterion, o2_criterion)
			backward(total_t_loss, total_o1_loss, total_o2_loss, num_examples)
			print("Train", str(epoch), "\t", total_t_loss.item() / num_examples["t"], "\t", correct_count / num_examples["t"])
			torch.save(self.model.state_dict(), self.save_name)
			validate(val_swt, t_criterion, o1_criterion, o2_criterion)
					
	def train_basic(self):

		def update(indices_to_use, option_indices, values, criterion):

			if len(indices_to_use) == 0:
				return

			update_input = torch.index_select(values, 0, indices_to_use).requires_grad_()
			update_target = torch.index_select(torch.tensor(option_indices, dtype=torch.long), 0, indices_to_use)

			self.optimizer.zero_grad()
			loss = criterion(update_input, update_target)
			if math.isnan(loss.item()):
				return loss.item()
			loss.backward(retain_graph=True)
			utils.clip_grad_norm_(self.model.parameters(), self.clip)
			self.optimizer.step()

			return loss.item()

		t_criterion = nn.NLLLoss()
		o1_criterion = nn.NLLLoss()
		o2_criterion = nn.NLLLoss()

		train_sampler = WeightedRandomSampler(self.get_weights(self.train_data), num_samples=len(self.train_data))
		train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, sampler=train_sampler, drop_last=True)

		for epoch in range(self.num_epochs):

			if epoch % 5 == 0:
				self.find_accuracy(self.val_data, epoch)

			for (states, instructions), actions in train_dataloader:

				t_indices_to_use = []
				o1_indices_to_use = []
				o2_indices_to_use = []

				template_idxs = []
				o1_idxs = []
				o2_idxs = []
				for i, action in enumerate(actions):
					template_idx, o1_idx, o2_idx = self.identify_components(action)
					reconstruction = self.template_to_string(template_idx, o1_idx, o2_idx)
					template_idxs.append(template_idx)
					o1_idxs.append(o1_idx)
					o2_idxs.append(o2_idx)
					if are_equivalent(reconstruction, action):
						t_indices_to_use.append(i)
						object_count = self.templates[template_idx].count("OBJ")
						if object_count >= 1:
							o1_indices_to_use.append(i)
						if object_count >= 2:
							o2_indices_to_use.append(i)

				q_ts, q_o1s, q_o2s = self.model(states, instructions)

				template_loss = update(torch.tensor(t_indices_to_use, dtype=torch.long), template_idxs, q_ts, t_criterion)
				print(epoch, "\t", template_loss, end="\t")

				if epoch >= self.o1_start:
					o1_loss = update(torch.tensor(o1_indices_to_use, dtype=torch.long), o1_idxs, q_o1s, o1_criterion)
					print(o1_loss, end="\t")
					if epoch >= self.o2_start:
						o2_loss = update(torch.tensor(o2_indices_to_use, dtype=torch.long), o2_idxs, q_o2s, o2_criterion)
						print(o2_loss, end="")

				print()

			if epoch % 10 == 0 and epoch != 0:
				torch.save(self.model.state_dict(), self.save_name)
				self.find_accuracy(self.train_data,  epoch)

		self.find_accuracy(self.val_data, self.num_epochs)
		self.find_accuracy(self.train_data, self.num_epochs)
		torch.save(self.model.state_dict(), self.save_name)

	def get_weights(self, data):

		sequential_sampler = SequentialSampler(data)
		base_dataloader = DataLoader(data, batch_size=1, sampler=sequential_sampler, drop_last=False)

		frequencies = np.zeros(len(self.templates))
		for _, action in base_dataloader:

			template_idx, o1_idx, o2_idx = self.identify_components(action[0])
			reconstruction = self.template_to_string(template_idx, o1_idx, o2_idx)
			if reconstruction == action[0]:
				frequencies[template_idx] += 1

		weights = np.zeros(len(data))
		for i, (_, action) in enumerate(base_dataloader):

			template_idx, o1_idx, o2_idx = self.identify_components(action[0])
			reconstruction = self.template_to_string(template_idx, o1_idx, o2_idx)
			if reconstruction == action[0]:
				if frequencies[template_idx] > 0:
					weights[i] = 1 / math.sqrt(frequencies[template_idx])

		return weights

	def find_accuracy(self, data, epoch=-1, print_examples=False):

		criterion = nn.NLLLoss()

		def update(condition, correct, total, guess, actual):
			if condition:
				if guess == actual:
					return correct + 1, total + 1
				else:
					return correct, total + 1
			return correct, total

		def get_loss(condition, loss_list, probabilities, correct):
			if condition:
				correct_idx_list = [correct[0]] if type(correct) == type([]) else [correct]
				loss_list.append(criterion(probabilities, torch.tensor(correct_idx_list, dtype=torch.long)).item())
			return loss_list

		def get_average_loss(loss_list):
			if len(loss_list) == 0:
				return
			return np.mean(np.asarray(loss_list))

		correct_templates = 0
		total_templates = 0
		correct_o1 = 0
		total_o1 = 0
		correct_o2 = 0
		total_o2 = 0

		t_losses = []
		o1_losses = []
		o2_losses = []

		template_counts = np.zeros(len(self.templates))
		ranks = []

		for i in range(len(data)):
			(state, instruction), action = data[i]
			template_truth, o1_truth, o2_truth = self.identify_components(action)
			reconstruction = self.template_to_string(template_truth, o1_truth, o2_truth)

			if are_equivalent(reconstruction, action):

				template_guess, o1_guess, o2_guess, t_prob, o1_prob, o2_prob = self.model.eval([state], [instruction])
				if print_examples:

					t_prob_list = []
					for i, template in enumerate(self.templates):
						t_prob_list.append(t_prob[0, i].item())
					sorted_ts = list(reversed(sorted(t_prob_list)))

					rank = -1
					for i, score in enumerate(sorted_ts):
						if t_prob[0, template_truth].item() == score:
							rank = i + 1

					print(self.template_to_string(template_guess, o1_guess, o2_guess), "\t", reconstruction, "\t", rank)
					ranks.append(rank)

					if template_counts[template_truth] == 0:
						template_counts[template_truth] = 1

				else:
					object_count = self.templates[template_truth].count("OBJ")
					correct_templates, total_templates = update(True, correct_templates, total_templates, template_guess, template_truth)
					correct_o1, total_o1 = update(object_count >= 1, correct_o1, total_o1, o1_guess, o1_truth)
					correct_o2, total_o2 = update(object_count >= 2, correct_o2, total_o2, o2_guess, o2_truth)

					t_losses = get_loss(True, t_losses, t_prob, template_truth)
					o1_losses = get_loss(object_count >= 1, o1_losses, o1_prob, o1_truth)
					o2_losses = get_loss(object_count >= 2, o2_losses, o2_prob, o2_truth)

		if not print_examples:
			if epoch >= 0:
				print("\nEPOCH", epoch)
			print("\tTemplates: \n\t\tAccuracy: ", correct_templates, "/", total_templates, "\n\t\tLoss", get_average_loss(t_losses))
			print("\tObject 1: \n\t\tAccuracy: ", correct_o1, "/", total_o1, "\n\t\tLoss", get_average_loss(o1_losses))
			print("\tObject 2: \n\t\tAccuracy", correct_o2, "/", total_o2, "\n\t\tLoss", get_average_loss(o2_losses))
			print()
		else:
			print(np.mean(np.asarray(ranks)), "/", np.sum(template_truth))

	def identify_components(self, action):

		lookarounds = ["with", "to", "from", "at", "in", "under", "into"]
		lookaheads = "".join("(?!\\b" + la + "\\b)" for la in lookarounds)
		lookbehinds = "".join("(?<!\\b" + la + "\\b)" for la in lookarounds)

		regex_string = (lookaheads + "(\w+(?:\s?\w+){0,3}?)" + lookbehinds).replace("\w", "(?:" + lookaheads + "\w)")

		def find_regular_match(template_string, idx, action_to_use):

			output = [-1, [], []]

			match_obj = re.fullmatch(template_string.replace("OBJ", regex_string), action_to_use)

			if match_obj != None:
				output[0] = idx
				for i in range(0, len(match_obj.groups())):
					word = match_obj.group(i + 1)
					for individual_word in word.split(" "):
						output[i + 1].append(self.reverse_vocab[individual_word[0:min(6, len(individual_word))].lower()])
				return output
			return None

		def edit_action(test_string, replace_cmd=None):
			removed_quotes =  "".join(test_string.split("\""))
			if replace_cmd != None:
				return removed_quotes.replace(replace_cmd, "CMD")
			return removed_quotes


		output = [-1, [], []]

		if "," in action:
			action = action[action.index(",") + 1:].strip(" ")

		for i, template_string in enumerate(self.templates):

			if "CMD" in template_string:
				first_word = template_string.split(" ")[0]
				quote_regex = re.compile("(?:" + first_word + ".*?(\"[\w|\s]+\"))|(?:" + first_word + ".*?to ([\w|\s]+))")
				quote_match = re.search(quote_regex, action)

				if quote_match != None:

					idx = 1 if quote_match.group(1) != None else 2
					inner_command = quote_match.group(idx).lstrip("\"").rstrip("\"")
					for j, inner_template in enumerate(self.templates):
						if "CMD" not in inner_template:
							inner_match = find_regular_match(inner_template, j, inner_command)
							if inner_match != None:
								replaced_match = find_regular_match(template_string, i, edit_action(action, inner_command))
								if replaced_match != None:
									return replaced_match

			regular_output = find_regular_match(template_string, i, edit_action(action))
			if regular_output != None:
				return regular_output

		return output
	
	def template_to_string(self, template_idx, o1, o2):

		def find_replacement_string(index_list):
			replacement_list = []
			for obj_num in index_list:
				replacement_list.append(self.vocab[int(obj_num)])
			return " ".join(replacement_list)

		"""copied from Hausknecht et al (2019)"""
		template_string = self.templates[template_idx]
		number_of_objects = template_string.count('OBJ')
		if number_of_objects == 0:
			return template_string
		elif number_of_objects == 1:
			return template_string.replace('OBJ', find_replacement_string(o1))
		else:
			return template_string.replace('OBJ', find_replacement_string(o1), 1)\
				.replace('OBJ', find_replacement_string(o2), 1)

	def visualize_attention(self):

		wt = self.data.walkthroughs[0]
		sentence_atts = [None]
		word_atts = [None]
		count = 0
		print(self.model.attender.time_attention.weight)
		print(self.model.attender.time_attention.bias)
		print(self.model.instruction_encoder.attender.time_attention.weight)
		print(self.model.instruction_encoder.attender.time_attention.bias)

		print()

		for instruction, state, action, start in wt:
			if count != 0 and start:
				break

			q_ts, q_o1s, q_o2s, sentence_atts, word_atts = self.model([state], [instruction], sentence_atts, word_atts)
			sa = sentence_atts.detach().numpy()
			for element in sa[0]:
				print(element, end="\t")
			print()
			count += 1


if __name__ == "__main__":


	args = {
		"embedding_size": 8, 
		"hidden_size": 128, 
		"spm_path": "./spm_models/unigram_8k.model", 
		"rom_path": ["../z-machine-games-master/jericho-game-suite/zork1.z5", "../z-machine-games-master/jericho-game-suite/zork2.z5", "../z-machine-games-master/jericho-game-suite/zork3.z5"], 
		"walkthrough_filename": ["../walkthroughs/zork_super_walkthrough", "../walkthroughs/zork2_super_walkthrough", "../walkthroughs/zork3_super_walkthrough"],
		"clip": 40,
		"batch_size": 64,
		"learning_rate": 0.004, # originally 0.001
		"num_epochs": 300,
		"o1_start": 150,
		"o2_start": 200,
		"temp_path": "../walkthroughs/additional_templates",
		"add_word_path": "../walkthroughs/additional_words",
		"max_number_of_sentences": 35,
		"max_number_of_words": 100
	}

	agent = Agent_Zork(args, model_type="time", save_name="./models/timemodel3.pt")
	#agent.visualize_attention()
	agent.train()
	#agent.find_accuracy(agent.train_data, print_examples=True)
	#agent.visualize_embeddings()





