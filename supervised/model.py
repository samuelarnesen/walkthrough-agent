import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sentencepiece as spm
from utils import *
from attender import BasicAttender, TimeAttender
from encoders import StateEncoder, InstructionEncoder

class BasicModel(nn.Module):

	def __init__(self, args):

		super(BasicModel, self).__init__()

		self.args = args
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.embeddings = nn.Embedding(args["vocab_size"], args["embedding_size"])
		self.instruction_encoder = InstructionEncoder(args["embedding_size"], args["hidden_size"], args["spm_path"])
		self.state_encoder = StateEncoder(args["embedding_size"], args["hidden_size"], args["spm_path"])
		self.attender = BasicAttender(args["hidden_size"])

		self.t_scorer = Scorer(args["hidden_size"] * 4, args["template_size"])
		self.o1_scorer = Scorer((args["hidden_size"] * 4) + args["template_size"], args["output_vocab_size"])
		self.o2_scorer = Scorer((args["hidden_size"] * 4) + args["template_size"] + args["output_vocab_size"], args["output_vocab_size"])

	def forward(self, state, instruction):

		encoded_state = self.state_encoder(self.embeddings, state)
		full_instruction_encoder_output = self.instruction_encoder(self.embeddings, instruction, encoded_state)
		attended_instruction = self.attender(full_instruction_encoder_output, encoded_state)

		q_t = self.t_scorer(attended_instruction, encoded_state)
		q_o1 = self.o1_scorer(attended_instruction, encoded_state, [q_t.detach()])
		q_o2 = self.o2_scorer(attended_instruction, encoded_state, [q_t.detach(), q_o1.detach()])

		return F.log_softmax(q_t, dim=1), F.log_softmax(q_o1, dim=1), F.log_softmax(q_o2, dim=1)

	def eval(self, state, instruction):
		with torch.no_grad():
			t_prob, o1_prob, o2_prob = self.forward(state, instruction)
			t, o1, o2 = torch.argmax(t_prob, dim=1).item(), torch.argmax(o1_prob, dim=1).item(), torch.argmax(o2_prob, dim=1).item()
			return t, o1, o2, t_prob, o1_prob, o2_prob

	def get_embedding(self, idx):
		return self.embeddings(torch.tensor(idx, dtype=torch.long))

	def get_name(self):
		return "basic"

class TimeModel(nn.Module):

	def __init__(self, args):

		super(TimeModel, self).__init__()

		self.args = args
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.embeddings = nn.Embedding(args["vocab_size"], args["embedding_size"])
		self.instruction_encoder = InstructionEncoder(args["embedding_size"], args["hidden_size"], args["spm_path"], basic=False)
		self.state_encoder = StateEncoder(args["embedding_size"], args["hidden_size"], args["spm_path"])
		self.attender = TimeAttender(args["hidden_size"], args["max_number_of_sentences"])

		self.t_scorer = Scorer(args["hidden_size"] * 4, args["template_size"])
		self.o1_scorer = Scorer((args["hidden_size"] * 4) + args["template_size"], args["output_vocab_size"])
		self.o2_scorer = Scorer((args["hidden_size"] * 4) + args["template_size"] + args["output_vocab_size"], args["output_vocab_size"])

	def forward(self, state, instruction, previous_sentence_attention, previous_word_attention):

		encoded_state = self.state_encoder(self.embeddings, state)

		full_instruction_encoder_output, word_weights, sentence_lengths = self.instruction_encoder(self.embeddings, instruction, encoded_state, previous_word_attention)
		attended_instruction, sentence_weights = self.attender(full_instruction_encoder_output, encoded_state, previous_sentence_attention, sentence_lengths)

		q_t = self.t_scorer(attended_instruction, encoded_state)
		q_o1 = self.o1_scorer(attended_instruction, encoded_state, [q_t.detach()])
		q_o2 = self.o2_scorer(attended_instruction, encoded_state, [q_t.detach(), q_o1.detach()])

		return F.log_softmax(q_t, dim=1), F.log_softmax(q_o1, dim=1), F.log_softmax(q_o2, dim=1), sentence_weights, word_weights

	def eval(self, state, instruction, previous_attention):
		with torch.no_grad():
			t_prob, o1_prob, o2_prob, sentence_weights = self.forward(state, instruction, previous_attention)
			t, o1, o2 = torch.argmax(t_prob, dim=1).item(), torch.argmax(o1_prob, dim=1).item(), torch.argmax(o2_prob, dim=1).item()
			return t, o1, o2, t_prob, o1_prob, o2_prob, sentence_weights

	def get_embedding(self, idx):
		return self.embeddings(torch.tensor(idx, dtype=torch.long))

	def get_name(self):
		return "time"


class Scorer(nn.Module):

	def __init__(self, input_size, output_size):

		super(Scorer, self).__init__()

		self.linear_scorer_1 = nn.Linear(input_size, input_size)
		self.linear_scorer_2 = nn.Linear(input_size, output_size)

	def forward(self, attended_instruction, state, other=[]):

		full_input = torch.cat([attended_instruction, state], dim=1)
		for other_vec in other:
			full_input = torch.cat([full_input, other_vec], dim=1)

		return self.linear_scorer_2(F.relu(self.linear_scorer_1(full_input)))






