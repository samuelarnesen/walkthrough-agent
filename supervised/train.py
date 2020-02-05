from parse_walkthrough import Walkthrough_Dataset
from model import BasicModel
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler

import numpy as np
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
import random, re, math


class Agent_Zork:

	def __init__(self, args, model_name=None, index_file_name=None):

		self.data = Walkthrough_Dataset(args["walkthrough_filename"], args["rom_path"], args["spm_path"])
		self.binding = jericho.load_bindings(args["rom_path"])
		self.template_generator = TemplateActionGenerator(self.binding)
		self.vocab, self.reverse_vocab = self.load_vocab(args["rom_path"])
		self.batch_size = args["batch_size"]
		self.num_epochs = args["num_epochs"]

		if index_file_name == None:
			shuffled_idxs = list(range(len(self.data)))
			random.shuffle(shuffled_idxs)
			train_idx_end = int(0.75 * len(self.data))
			val_idx_end = int(0.875 * len(self.data))
			self.train_data = self.data.split(shuffled_idxs[:train_idx_end])
			self.val_data = self.data.split(shuffled_idxs[train_idx_end:val_idx_end])
			self.test_data = self.data.split(shuffled_idxs[val_idx_end:])
			self.dump_indices("data_indices", shuffled_idxs[:train_idx_end], shuffled_idxs[train_idx_end:val_idx_end], shuffled_idxs[val_idx_end:])
		else:
			train_idxs, val_idxs, test_idxs = self.load_indices(index_file_name)
			self.train_data = self.data.split(train_idxs)
			self.val_data = self.data.split(val_idxs)
			self.test_data = self.data.split(test_idxs)

		model_args = {
			"embedding_size": args["embedding_size"],
			"hidden_size": args["hidden_size"],
			"template_size": len(self.template_generator.templates),
			"vocab_size": self.data.get_vocab_size(),
			"output_vocab_size": len(self.vocab),
			"num_samples": 512,
			"batch_size": args["batch_size"]
		}
		self.model = BasicModel(model_args)
		self.optimizer = optim.Adam(self.model.parameters(), lr=args["learning_rate"]) 

		if model_name != None:
			self.model.load_state_dict(torch.load(model_name))

	def train(self):

		def update(indices_to_use, option_indices, values, criterion):

			if len(indices_to_use) == 0:
				return

			update_input = torch.index_select(values, 0, indices_to_use).requires_grad_()
			update_target = torch.index_select(torch.tensor(option_indices, dtype=torch.long), 0, indices_to_use)

			self.optimizer.zero_grad()
			loss = criterion(update_input, update_target)
			loss.backward(retain_graph=True)
			self.optimizer.step()

			return loss.item()

		t_criterion = nn.NLLLoss()
		o1_criterion = nn.NLLLoss()
		o2_criterion = nn.NLLLoss()

		#train_sampler = RandomSampler(self.train_data)
		train_sampler = WeightedRandomSampler(self.get_weights(self.train_data), num_samples=len(self.train_data))
		train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, sampler=train_sampler, drop_last=True)

		for epoch in range(self.num_epochs):

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
					if reconstruction == action:
						t_indices_to_use.append(i)
						object_count = self.template_generator.templates[template_idx].count("OBJ")
						if object_count >= 1:
							o1_indices_to_use.append(i)
						if object_count >= 2:
							o2_indices_to_use.append(i)

				q_ts, q_o1s, q_o2s = self.model(states, instructions)

				template_loss = update(torch.tensor(t_indices_to_use, dtype=torch.long), template_idxs, q_ts, t_criterion)
				o1_loss = update(torch.tensor(o1_indices_to_use, dtype=torch.long), o1_idxs, q_o1s, o1_criterion)
				o2_loss = update(torch.tensor(o2_indices_to_use, dtype=torch.long), o2_idxs, q_o2s, o2_criterion)

				print(epoch, "\t", template_loss, "\t", o1_loss, "\t", o2_loss)

			if epoch % 50 == 0 and epoch != 0:
				torch.save(self.model.state_dict(), "basic_model.pt")

		self.find_accuracy(self.val_data, self.num_epochs)
		torch.save(self.model.state_dict(), "basic_model.pt")

	def get_weights(self, data):

		sequential_sampler = SequentialSampler(data)
		base_dataloader = DataLoader(data, batch_size=1, sampler=sequential_sampler, drop_last=False)

		frequencies = np.zeros(len(self.template_generator.templates))
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
				loss_list.append(criterion(probabilities, torch.tensor([correct], dtype=torch.long)).item())
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

		template_counts = np.zeros(len(self.template_generator.templates))
		ranks = []

		for i in range(len(data)):
			(state, instruction), action = data[i]
			template_truth, o1_truth, o2_truth = self.identify_components(action)
			reconstruction = self.template_to_string(template_truth, o1_truth, o2_truth)

			if reconstruction == action:

				template_guess, o1_guess, o2_guess, t_prob, o1_prob, o2_prob = self.model.eval(state.unsqueeze(dim=0), instruction.unsqueeze(dim=0))
				if print_examples:

					t_prob_list = []
					for i, template in enumerate(self.template_generator.templates):
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
					object_count = self.template_generator.templates[template_truth].count("OBJ")
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
		output = [-1, -1, -1]
		for i, template_string in enumerate(self.template_generator.templates):
			match_obj = re.fullmatch(template_string.replace("OBJ", "(\w+)"), action)
			if match_obj != None:

				output[0] = i
				for i in range(0, len(match_obj.groups())):
					word = match_obj.group(i + 1)
					output[i + 1] = self.reverse_vocab[word[0:min(6, len(word))]]

				return output

		return output
	
	def template_to_string(self, template_idx, o1, o2):
		"""copied from Hausknecht et al (2019)"""
		template_string = self.template_generator.templates[template_idx]
		number_of_objects = template_string.count('OBJ')
		if number_of_objects == 0:
			return template_string
		elif number_of_objects == 1:
			return template_string.replace('OBJ', self.vocab[int(o1)])
		else:
			return template_string.replace('OBJ', self.vocab[int(o1)], 1)\
				.replace('OBJ', self.vocab[int(o2)], 1)

	def load_vocab(self, rom_path):
		"""copied from Hausknecht et al (2019)"""
		env = FrotzEnv(rom_path)
		vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
		reverse = {vocab[w]: w for w in vocab}
		return vocab, reverse

	def load_indices(self, file_path):
		train_idxs = []
		val_idxs = []
		test_idxs = []
		with open(file_path) as f:
			current = ""
			for line in f.readlines():
				if "train" in line or "val" in line or "test" in line:
					current = line.strip(" \n:")
					continue
				if current == "train":
					train_idxs.append(int(line.strip(" \n")))
				elif current == "val":
					val_idxs.append(int(line.strip(" \n")))
				elif current == "test":
					test_idxs.append(int(line.strip(" \n")))
		return train_idxs, val_idxs, test_idxs

	def dump_indices(self, file_path, train_idxs, val_idxs, test_idxs):

		with open(file_path, "w") as f:
			print("train", file=f)
			for idx in train_idxs:
				print(idx, file=f)

			print("val", file=f)
			for idx in val_idxs:
				print(idx, file=f)

			print("test", file=f)
			for idx in test_idxs:
				print(idx, file=f)


if __name__ == "__main__":


	args = {
		"embedding_size": 64, 
		"hidden_size": 128, 
		"spm_path": "./spm_models/unigram_8k.model", 
		"rom_path": "zork1.z5", 
		"walkthrough_filename": "../walkthroughs/zork_super_walkthrough",
		"steps": 1000000,
		"max_seq_len": 250,
		"batch_size": 16,
		"learning_rate": 0.0005,
		"num_samples": 512,
		"num_epochs": 80,
	}

	agent = Agent_Zork(args, model_name="basic_model.pt", index_file_name="data_indices")
	#agent.train()
	agent.find_accuracy(agent.train_data, print_examples=True)





