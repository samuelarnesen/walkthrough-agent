from parse_walkthrough import Walkthrough_Dataset
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
import random, re, math

class Agent_Zork:

	def __init__(self, args, model_name=None, index_file_name=None, save_name="basic_model.pt"):

		self.data = Walkthrough_Dataset(args["walkthrough_filename"], args["rom_path"])
		self.templates = create_templates_list(args["rom_path"], args["temp_path"])
		self.vocab, self.reverse_vocab = load_vocab(args["rom_path"], args["add_word_path"])
		self.batch_size = args["batch_size"]
		self.num_epochs = args["num_epochs"]
		self.clip = args["clip"]
		self.learning_rate = args["learning_rate"]
		self.save_name = save_name
		self.o1_start = args["o1_start"]
		self.o2_start = args["o2_start"]

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
			"output_vocab_size": len(self.vocab),
			"batch_size": args["batch_size"],
			"max_number_of_sentences": args["max_number_of_sentences"]
		}
		self.model = BasicModel(self.model_args)
		self.optimizer = optim.Adam(self.model.parameters(), lr=args["learning_rate"]) 

		if model_name != None:
			self.model.load_state_dict(torch.load(model_name))

	def train(self, method="batch"):
		if method == "batch":
			self.train_batch()
		elif method == "sequential":
			self.train_sequential()
		else:
			print("METHOD NOT AVAILABLE")

	def train_sequential(self):

		self.model = TimeModel(self.model_args)
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
		t_criterion = nn.NLLLoss()

		for epoch in range(5):

			for wt_num, wt in enumerate(self.data.walkthroughs):
				t_loss = torch.zeros([1])
				a_dists = []
				num_examples = 0
				for pair in wt.section_generator():
					# pair set up is instruction, state, action, start
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
							o1_idxs.append(o1_idx)
							o2_idxs.append(o2_idx)
							if section[3]:
								a_dists.append(None)
						else:
							sections_missing.append(i)

					a_dists = torch.index_select(a_dists, 0, torch.tensor(sections_used, dtype=torch.long)) if torch.is_tensor(a_dists) else a_dists
					q_ts, q_o1s, q_o2s, a_dists = self.model(states, instructions, a_dists)

					t_loss = torch.add(t_loss, torch.mul(t_criterion(q_ts, torch.tensor(template_idxs, dtype=torch.long)), len(sections_used)))
					num_examples += len(sections_used)

					for i in sections_missing:
						a_dists = torch.cat([a_dists[0:i, :], torch.zeros([1, self.model_args["max_number_of_sentences"]]), a_dists[i:, :]])

				t_loss = torch.div(t_loss, num_examples)
				print(str(epoch) + "." + str(wt_num), "\t", t_loss.item())
				self.optimizer.zero_grad()
				t_loss.backward(retain_graph=True)
				utils.clip_grad_norm_(self.model.parameters(), self.clip)
				self.optimizer.step()

			torch.save(self.model.state_dict(), self.save_name)
					

	def train_batch(self):

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

		def find_regular_match(template_string, idx, action_to_use):

			output = [-1, [], []]
			match_obj = re.fullmatch(template_string.replace("OBJ", "(\w+(?:\s?\w+){0,3}?)"), action_to_use)

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

if __name__ == "__main__":


	args = {
		"embedding_size": 8, 
		"hidden_size": 128, 
		"spm_path": "./spm_models/unigram_8k.model", 
		"rom_path": ["../z-machine-games-master/jericho-game-suite/zork1.z5", "../z-machine-games-master/jericho-game-suite/zork2.z5", "../z-machine-games-master/jericho-game-suite/zork3.z5"], 
		"walkthrough_filename": ["../walkthroughs/zork_super_walkthrough", "../walkthroughs/zork2_super_walkthrough", "../walkthroughs/zork3_super_walkthrough"],
		"clip": 40,
		"max_seq_len": 250,
		"batch_size": 64,
		"learning_rate": 0.0005,
		"num_epochs": 300,
		"o1_start": 150,
		"o2_start": 200,
		"temp_path": "../walkthroughs/additional_templates",
		"add_word_path": "../walkthroughs/additional_words",
		"max_number_of_sentences": 35
	}

	agent = Agent_Zork(args, save_name="./models/timemodel0.pt")
	agent.train(method="sequential")
	#agent.find_accuracy(agent.train_data, print_examples=True)
	#agent.visualize_embeddings()





