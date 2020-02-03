import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class BasicModel(nn.Module):

	def __init__(self, args):

		super(BasicModel, self).__init__()

		self.args = args
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.embeddings = nn.Embedding(args["vocab_size"], args["embedding_size"])
		self.instruction_encoder = Instruction_Encoder(args["embedding_size"], args["hidden_size"])
		self.state_encoder = State_Encoder(args["embedding_size"], args["hidden_size"])

		self.t_scorer = nn.Linear(args["hidden_size"] * 4, args["template_size"])
		self.o1_scorer = nn.Linear(args["hidden_size"] * 4, args["output_vocab_size"])
		self.o2_scorer = nn.Linear(args["hidden_size"] * 4, args["output_vocab_size"])

	def forward(self, state, instruction):

		encoded_instruction = self.instruction_encoder(self.embeddings(instruction))
		encoded_state = self.state_encoder(self.embeddings(state))

		full_input = torch.cat([encoded_instruction, encoded_state], dim=1)

		q_t = self.t_scorer(full_input)
		q_o1 = self.o1_scorer(full_input)
		q_o2 = self.o2_scorer(full_input)

		return F.log_softmax(q_t, dim=1), F.log_softmax(q_o1, dim=1), F.log_softmax(q_o2, dim=1)

	def act(self, state, instruction):
		with torch.no_grad():
			t_prob, o1_prob, o2_prob = self.forward(state, instruction)
			t, o1, o2 = torch.argmax(t_prob, dim=1).item(), torch.argmax(o1_prob, dim=1).item(), torch.argmax(o2_prob, dim=1).item()
			return t, o1, o2, t_prob, o1_prob, o2_prob

	def flatten_parameters(self):
		self.state_encoder.flatten_parameters()
		self.instruction_encoder.flatten_parameters()

class State_Encoder(nn.Module):

	def __init__(self, embedding_size, hidden_size):

		super(State_Encoder, self).__init__()

		self.encoder_zero = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
		self.encoder_one = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
		self.encoder_two = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
		self.encoder_three = nn.LSTM(embedding_size, hidden_size, bidirectional=True)

		self.combiner = nn.Linear(hidden_size * 8, hidden_size * 2)

	def forward(self, state):

		encoded_zero, _ = self.encoder_zero(state[:, 0, :, :].permute(1, 0, 2))
		encoded_one, _ = self.encoder_one(state[:, 1, :, :].permute(1, 0, 2))
		encoded_two, _ = self.encoder_two(state[:, 2, :, :].permute(1, 0, 2))
		encoded_three, _ = self.encoder_three(state[:, 3, :, :].permute(1, 0, 2))

		combined_tensor = torch.cat([encoded_zero[-1, :, :], encoded_one[-1, :, :], encoded_two[-1, :, :], encoded_three[-1, :, :]], dim=1)
		linear_combo = self.combiner(combined_tensor)

		return linear_combo

	def flatten_parameters(self):

		self.encoder_zero.flatten_parameters()
		self.encoder_one.flatten_parameters()
		self.encoder_two.flatten_parameters()
		self.encoder_three.flatten_parameters()

class Instruction_Encoder(nn.Module):

	def __init__(self, embedding_size, hidden_size):

		super(Instruction_Encoder, self).__init__()
		self.encoder = nn.LSTM(embedding_size, hidden_size, bidirectional=True)

	def forward(self, instruction):

		temp_instruction = instruction.squeeze(dim=1).permute(1, 0, 2)
		output, _ = self.encoder(temp_instruction)
		return output[-1, :, :]

	def flatten_parameters(self):

		self.encoder.flatten_parameters()

