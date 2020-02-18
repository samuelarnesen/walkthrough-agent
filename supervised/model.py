import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sentencepiece as spm
import nltk.data
from utils import *
from attender import BasicAttender


class BasicModel(nn.Module):

	def __init__(self, args):

		super(BasicModel, self).__init__()

		self.args = args
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.embeddings = nn.Embedding(args["vocab_size"], args["embedding_size"])
		self.instruction_encoder = Instruction_Encoder(args["embedding_size"], args["hidden_size"], args["spm_path"])
		self.state_encoder = State_Encoder(args["embedding_size"], args["hidden_size"], args["spm_path"])
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

	def flatten_parameters(self):
		self.state_encoder.flatten_parameters()
		self.instruction_encoder.flatten_parameters()

	def get_embedding(self, idx):
		return self.embeddings(torch.tensor(idx, dtype=torch.long))

class State_Encoder(nn.Module):

	def __init__(self, embedding_size, hidden_size, spm_path):

		super(State_Encoder, self).__init__()

		self.encoder_zero = nn.LSTM(embedding_size, int(hidden_size / 2), bidirectional=True)
		self.encoder_one = nn.LSTM(embedding_size, int(hidden_size / 4), bidirectional=True)
		self.encoder_two = nn.LSTM(embedding_size, int(hidden_size / 8), bidirectional=True)
		self.encoder_three = nn.LSTM(embedding_size, int(hidden_size / 8), bidirectional=True)

		self.sp = spm.SentencePieceProcessor()
		self.sp.Load(spm_path)

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def forward(self, embeddings, state_text):

		state = embeddings(convert_batch_to_tokens(state_text, 350, self.device, self.sp))

		encoded_zero, _ = self.encoder_zero(state[:, 0, :, :].permute(1, 0, 2))
		encoded_one, _ = self.encoder_one(state[:, 1, :, :].permute(1, 0, 2))
		encoded_two, _ = self.encoder_two(state[:, 2, :, :].permute(1, 0, 2))
		encoded_three, _ = self.encoder_three(state[:, 3, :, :].permute(1, 0, 2))

		combined_tensor = torch.cat([encoded_zero[-1, :, :], encoded_one[-1, :, :], encoded_two[-1, :, :], encoded_three[-1, :, :]], dim=1)
		return combined_tensor

	def flatten_parameters(self):

		self.encoder_zero.flatten_parameters()
		self.encoder_one.flatten_parameters()
		self.encoder_two.flatten_parameters()
		self.encoder_three.flatten_parameters()

class Instruction_Encoder(nn.Module):

	def __init__(self, embedding_size, hidden_size, spm_path):

		super(Instruction_Encoder, self).__init__()
		self.word_encoder = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
		self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.sp = spm.SentencePieceProcessor()
		self.sp.Load(spm_path)

		self.hidden_size = hidden_size
		self.attender = BasicAttender(self.hidden_size)

	def forward(self, embeddings, instructions, encoded_state, max_sentence_number=35):

		def encode_instruction(instruction, index):
			sentence_list = self.tokenizer.tokenize(instruction)
			sentence_tensor = convert_batch_to_tokens(sentence_list, 100, self.device, self.sp)
			embedded_sentence_tensor = embeddings(sentence_tensor).squeeze(1).permute(1, 0, 2)
			sentence_encoder_output, _ = self.word_encoder(embedded_sentence_tensor)

			encoded_instruction = self.attender(sentence_encoder_output, encoded_state[index, :])

			return torch.cat([encoded_instruction, torch.zeros(max_sentence_number - len(sentence_list), self.hidden_size * 2)], dim=0)

		encoded_instructions = encode_instruction(instructions[0], 0).unsqueeze(0)
		for i in range(1, len(instructions)):
			encoded_instructions = torch.cat([encoded_instructions, encode_instruction(instructions[i], i).unsqueeze(0)], dim=0)

		return encoded_instructions.permute(1, 0, 2)

	def flatten_parameters(self):
		self.encoder.flatten_parameters()

class Scorer(nn.Module):

	def __init__(self, input_size, output_size):

		super(Scorer, self).__init__()

		self.linear_scorer_1 = nn.Linear(input_size, input_size)
		self.linear_scorer_2 = nn.Linear(input_size, output_size)

	def forward(self, attended_instruction, state, other=[]):

		full_input = torch.cat([attended_instruction, state], dim=1)
		for other_vec in other:
			full_input = torch.cat([full_input, other_vec], dim=1)

		return self.linear_scorer_2(self.linear_scorer_1(full_input))






