from parse_walkthrough import Walkthrough_Dataset
from model import BasicModel
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler

import numpy as np
import random
import re


class Agent_Zork:

	def __init__(self, args):

		self.data = Walkthrough_Dataset(args["walkthrough_filename"], args["rom_path"], args["spm_path"])
		self.binding = jericho.load_bindings(args["rom_path"])
		self.template_generator = TemplateActionGenerator(self.binding)
		self.vocab, self.reverse_vocab = self.load_vocab(args["rom_path"])
		self.batch_size = args["batch_size"]
		self.num_epochs = args["num_epochs"]

		shuffled_idxs = list(range(len(self.data)))
		random.shuffle(shuffled_idxs)
		train_idx_end = int(0.75 * len(self.data))
		val_idx_end = int(0.875 * len(self.data))
		self.train_data = self.data.split(shuffled_idxs[:train_idx_end])
		self.val_data = self.data.split(shuffled_idxs[train_idx_end:val_idx_end])
		self.test_data = self.data.split(shuffled_idxs[val_idx_end:])

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

		train_sampler = RandomSampler(self.train_data)
		train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, sampler=train_sampler, drop_last=True)

		for epoch in range(self.num_epochs):

			self.find_validation_accuracy(epoch)

			for (states, instructions), actions in train_dataloader:

				q_ts, q_o1s, q_o2s = self.model(states, instructions)

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

				template_loss = update(torch.tensor(t_indices_to_use, dtype=torch.long), template_idxs, q_ts, t_criterion)
				o1_loss = update(torch.tensor(o1_indices_to_use, dtype=torch.long), o1_idxs, q_o1s, o1_criterion)
				o2_loss = update(torch.tensor(o2_indices_to_use, dtype=torch.long), o2_idxs, q_o2s, o2_criterion)

			if epoch % 50 == 0:
				torch.save(self.model.state_dict(), "basic_model.pt")

		self.find_validation_accuracy(self.num_epochs)


	def find_validation_accuracy(self, epoch=-1):

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

		for i in range(len(self.val_data)):
			(state, instruction), action = self.val_data[i]
			template_truth, o1_truth, o2_truth = self.identify_components(action)
			reconstruction = self.template_to_string(template_truth, o1_truth, o2_truth)

			if reconstruction == action:
				template_guess, o1_guess, o2_guess, t_prob, o1_prob, o2_prob = self.model.act(state.unsqueeze(dim=0), instruction.unsqueeze(dim=0))
				object_count = self.template_generator.templates[template_truth].count("OBJ")
				correct_templates, total_templates = update(True, correct_templates, total_templates, template_guess, template_truth)
				correct_o1, total_o1 = update(object_count >= 1, correct_o1, total_o1, o1_guess, o1_truth)
				correct_o2, total_o2 = update(object_count >= 2, correct_o2, total_o2, o2_guess, o2_truth)

				t_losses = get_loss(True, t_losses, t_prob, template_truth)
				o1_losses = get_loss(object_count >= 1, o1_losses, o1_prob, o1_truth)
				o2_losses = get_loss(object_count >= 2, o2_losses, o2_prob, o2_truth)

		if epoch >= 0:
			print("EPOCH", epoch)
		print("\tTemplates: \n\t\tAccuracy: ", correct_templates, "/", total_templates, "\n\t\tLoss", get_average_loss(t_losses))
		print("\tObject 1: \n\t\tAccuracy: ", correct_o1, "/", total_o1, "\n\t\tLoss", get_average_loss(o1_losses))
		print("\tObject 2: \n\t\tAccuracy", correct_o2, "/", total_o2, "\n\t\tLoss", get_average_loss(o2_losses))
		print()


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
		"num_epochs": 300,
	}

	agent = Agent_Zork(args)
	agent.train()





