import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import nltk.data
import sentencepiece as spm
from utils import *
from attender import BasicAttender

class StateEncoder(nn.Module):

	def __init__(self, embedding_size, hidden_size, spm_path):

		super(StateEncoder, self).__init__()

		self.encoder_zero = nn.LSTM(embedding_size, int(hidden_size / 2), bidirectional=True)
		self.encoder_one = nn.LSTM(embedding_size, int(hidden_size / 4), bidirectional=True)
		self.encoder_two = nn.LSTM(embedding_size, int(hidden_size / 8), bidirectional=True)
		self.encoder_three = nn.LSTM(embedding_size, int(hidden_size / 8), bidirectional=True)

		self.sp = spm.SentencePieceProcessor()
		self.sp.Load(spm_path)

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def forward(self, embeddings, state_text):

		state = embeddings(convert_batch_to_tokens(state_text, 420, self.device, self.sp))

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

class InstructionEncoder(nn.Module):

	def __init__(self, embedding_size, hidden_size, spm_path):

		super(InstructionEncoder, self).__init__()
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