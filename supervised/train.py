from parse_walkthrough import Walkthrough_Dataset, SuperWalkthrough
from utils import *
from models import BasicModel, TimeModel, TransformerModel
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
import random, re, math, sys

class Agent_Zork:

	def __init__(self, args, model_type="basic", model_name=None, index_file_name=None, save_name="./models/basic_model.pt"):

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

		self.model = None
		if model_type == "basic":
			self.model = BasicModel(self.model_args).to(self.device)
		elif model_type == "time":
			self.model = TimeModel(self.model_args).to(self.device)
		elif model_type == "transformer":
			self.model = TransformerModel(self.model_args).to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=args["learning_rate"])

		if model_name != None:
			self.model.load_state_dict(torch.load(model_name))

	def train(self):
		"""
		trains and validates the model
		"""
		if self.model.get_name() == "basic" or self.model.get_name() == "transformer":
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

		def get_class_weights(swt):
			"""
			get weights for each class
			"""

			t_freq = torch.zeros(len(self.templates))
			o1_freq = torch.zeros(len(self.vocab) + 2)
			o2_freq = torch.zeros(len(self.vocab) + 2)

			for _, _, action, _ in swt:
				t_idx, o1_seq, o2_seq = self.identify_components(action)
				t_freq[t_idx] +=1
				if len(o1_seq) > 0:
					o1_freq[get_object_index(o1_seq)] += 1
					if len(o2_seq) > 0:
						o2_freq[get_object_index(o2_seq)] += 1

			t_weight, o1_weight, o2_weight = torch.zeros(len(self.templates)), torch.zeros(len(self.vocab) + 2), torch.zeros(len(self.vocab) + 2)
			for i in range(len(self.templates)):
				t_weight[i] = 1 / (math.sqrt(t_freq[i]) if t_freq[i] > 0 else 1)
			for i in range(len(self.vocab) + 2):
				o1_weight[i] = 1 / (math.sqrt(o1_freq[i]) if o1_freq[i] > 0 else 1)
				o2_weight[i] = 1 / (math.sqrt(o2_freq[i]) if o2_freq[i] > 0 else 1)

			return t_weight, o1_weight, o2_weight

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
			for parameter, data in self.model.named_parameters():
				if "attender.initial" in parameter:
					data = data.clamp(0, float("inf"))
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
		train_swt = SuperWalkthrough(self.walkthrough_filenames[0:-1], self.rom_paths[0:-1])
		val_swt = SuperWalkthrough([self.walkthrough_filenames[-1]], [self.rom_paths[-1]])

		t_weight, o1_weight, o2_weight = get_class_weights(train_swt)

		t_criterion = nn.NLLLoss(weight=t_weight, reduction='sum')
		o1_criterion = nn.NLLLoss(weight=o1_weight, reduction='sum')
		o2_criterion = nn.NLLLoss(weight=o2_weight, reduction='sum')

		validate(val_swt, t_criterion, o1_criterion, o2_criterion)


		for epoch in range(self.num_epochs):
			total_t_loss, total_o1_loss, total_o2_loss, num_examples, correct_count = execute_epoch(train_swt, t_criterion, o1_criterion, o2_criterion)
			backward(total_t_loss, total_o1_loss, total_o2_loss, num_examples)
			print("Train", str(epoch), "\t", total_t_loss.item() / num_examples["t"], "\t", correct_count / num_examples["t"])
			torch.save(self.model.state_dict(), self.save_name)
			validate(val_swt, t_criterion, o1_criterion, o2_criterion)
					
	def train_basic(self):

		def update(indices_to_use, option_indices, values, criterion, retain=False):

			if len(indices_to_use) == 0:
				return

			update_input = torch.index_select(values, 0, indices_to_use).requires_grad_()
			update_target = torch.index_select(torch.tensor(option_indices, dtype=torch.long), 0, indices_to_use)

			loss = criterion(update_input, update_target)

			if math.isnan(loss.item()):
				sys.exit()
			loss.backward(retain_graph=retain)

			self.optimizer.step()
			self.optimizer.zero_grad()

			return loss.item()


		t_criterion = nn.NLLLoss()
		o1_criterion = nn.NLLLoss()
		o2_criterion = nn.NLLLoss()

		train_sampler = WeightedRandomSampler(self.get_weights(self.train_data), num_samples=len(self.train_data))
		train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, sampler=train_sampler, drop_last=True)
		#train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=False)


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

				template_loss = update(torch.tensor(t_indices_to_use, dtype=torch.long), template_idxs, q_ts, t_criterion, retain=(epoch >= self.o1_start))
				print(epoch, "\t", template_loss, end="\t")

				if epoch >= self.o1_start:
					o1_loss = update(torch.tensor(o1_indices_to_use, dtype=torch.long), o1_idxs, q_o1s, o1_criterion, retain=(epoch >= self.o2_start))
					print(o1_loss, end="\t")
					if epoch >= self.o2_start:
						o2_loss = update(torch.tensor(o2_indices_to_use, dtype=torch.long), o2_idxs, q_o2s, o2_criterion, retain=False)
						print(o2_loss, end="")

				print()

			#if epoch % 10 == 0 and epoch != 0:
			torch.save(self.model.state_dict(), self.save_name)
			self.find_accuracy(self.val_data, epoch)

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
		states = []
		instructions = []
		template_truths = []

		for (state, instruction), action in data:
			states.append(state)
			instructions.append(instruction)
			template_truth, _, _ = self.identify_components(action)
			template_truths.append(template_truth)


		t_guesses, o1_guesses, o2_guesses, t_prob, o1_prob, o2_prob = self.model.eval(states, instructions)
		loss = criterion(t_prob, torch.tensor(template_truths, dtype=torch.long))

		correct = 0
		total = 0
		for i, guess in enumerate(t_guesses):
			if guess == template_truths[i]:
				correct += 1
			total += 1
		print("\t\tAccuracy:", correct, "/", total, "\n\t\tLoss:", loss.item(), "\n")

	
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
		"learning_rate": 0.001, # originally 0.001
		"num_epochs": 250,
		"o1_start": 15000,
		"o2_start": 20000,
		"temp_path": "../walkthroughs/additional_templates",
		"add_word_path": "../walkthroughs/additional_words",
		"max_number_of_sentences": 35,
		"max_number_of_words": 100
	}

	agent = Agent_Zork(args, model_type="transformer", save_name="./models/transformer_model.pt")
	#agent.visualize_attention()
	agent.train()
	#agent.find_accuracy(agent.train_data, print_examples=True)
	#agent.visualize_embeddings()
