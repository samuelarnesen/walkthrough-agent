import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.autograd import Variable
import nltk.data
import sentencepiece as spm
from utils import *
from attender import BasicAttender, TimeAttender
import sys

class StateEncoder(nn.Module):

	def __init__(self, embedding_size, hidden_size, spm_path):

		super(StateEncoder, self).__init__()

		self.encoder_zero = nn.LSTM(embedding_size, int(hidden_size / 2), bidirectional=True, batch_first=True)
		self.encoder_one = nn.LSTM(embedding_size, int(hidden_size / 4), bidirectional=True, batch_first=True)
		self.encoder_two = nn.LSTM(embedding_size, int(hidden_size / 8), bidirectional=True, batch_first=True)
		self.encoder_three = nn.LSTM(embedding_size, int(hidden_size / 8), bidirectional=True, batch_first=True)

		self.sp = spm.SentencePieceProcessor()
		self.sp.Load(spm_path)

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def forward(self, embeddings, state_text):

		tokens = convert_batch_to_tokens(state_text, 420, self.device, self.sp, embeddings)
		outputs = []
		for i, encoder in enumerate([self.encoder_zero, self.encoder_one, self.encoder_two, self.encoder_three]):
			encoded, _ = encoder(tokens[i])
			unpacked, _ = pad_packed_sequence(encoded)
			outputs.append(unpacked[-1, :, :])

		combined_tensor = torch.cat(outputs, dim=1)

		return combined_tensor

	def flatten_parameters(self):

		self.encoder_zero.flatten_parameters()
		self.encoder_one.flatten_parameters()
		self.encoder_two.flatten_parameters()
		self.encoder_three.flatten_parameters()

class InstructionEncoder(nn.Module):

	def __init__(self, embedding_size, hidden_size, spm_path, basic=True, max_word_number=100):

		super(InstructionEncoder, self).__init__()
		self.basic = basic
		self.word_encoder = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
		self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.sp = spm.SentencePieceProcessor()
		self.sp.Load(spm_path)

		self.hidden_size = hidden_size
		self.attender = BasicAttender(self.hidden_size) if basic else TimeAttender(self.hidden_size, max_word_number)

	def forward(self, embeddings, instructions, encoded_state, previous_attention=None, max_sentence_number=35, max_word_number=100):

		if self.basic:
			return self.basic_encode(embeddings, instructions, encoded_state, max_sentence_number, max_word_number)
		else:
			return self.time_encode(embeddings, instructions, encoded_state, max_sentence_number, max_word_number, previous_attention)

	def basic_encode(self, embeddings, instructions, encoded_state, max_sentence_number, max_word_number):

		def encode_instruction(instruction, index):
			sentence_list = self.tokenizer.tokenize(instruction)
			sentence_tensor = convert_batch_to_tokens(sentence_list, max_word_number, self.device, self.sp, embeddings)
			sentence_encoder_output, _ = self.word_encoder(sentence_tensor)
			unpacked, _ = pad_packed_sequence(sentence_encoder_output)


			# unpacked are --max length x num_sentences x 256---
			encoded_instruction = self.attender(unpacked, encoded_state[index, :])

			return torch.cat([encoded_instruction, torch.zeros(max_sentence_number - len(sentence_list), self.hidden_size * 2)], dim=0).unsqueeze(0)

		encoded_instructions = encode_instruction(instructions[0], 0)
		for i in range(1, len(instructions)):
			encoded_instructions = torch.cat([encoded_instructions, encode_instruction(instructions[i], i)], dim=0)

		return encoded_instructions.permute(1, 0, 2)

	def time_encode(self, embeddings, instructions, encoded_state, max_sentence_number, max_word_number, previous_attention):

		number_of_sentences = []

		def get_lengths_of_sentences(sentence_list):
			lengths = []
			for sentence in sentence_list:
				lengths.append(len(sentence.split(" ")))
			return lengths

		def encode_instruction(instruction, index):
			sentence_list = self.tokenizer.tokenize(instruction)
			sentence_tensor = convert_batch_to_tokens(sentence_list, max_word_number, self.device, self.sp)
			embedded_sentence_tensor = embeddings(sentence_tensor).squeeze(1).permute(1, 0, 2)
			sentence_encoder_output, _ = self.word_encoder(embedded_sentence_tensor)

			attention_to_use = previous_attention[index, 0:len(sentence_list), :] if torch.is_tensor(previous_attention) else [None]
			encoded_instruction, attention = self.attender(sentence_encoder_output[:, :, :], encoded_state[index, :], attention_to_use, get_lengths_of_sentences(sentence_list))

			encoded_instruction = torch.cat([encoded_instruction, torch.zeros(max_sentence_number - len(sentence_list), self.hidden_size * 2)], dim=0).unsqueeze(0)
			attention = torch.cat([attention, torch.zeros(max_sentence_number - len(sentence_list), max_word_number)]).unsqueeze(0)

			number_of_sentences.append(len(sentence_list))

			return encoded_instruction, attention

		encoded_instructions, attentions = encode_instruction(instructions[0], 0)
		for i in range(1, len(instructions)):
			next_encoded_instructions, next_attentions = encode_instruction(instructions[i], i)
			encoded_instructions = torch.cat([encoded_instructions, next_encoded_instructions], dim=0)
			attentions = torch.cat([attentions, next_attentions], dim=0)

		return encoded_instructions.permute(1, 0, 2), attentions, number_of_sentences

	def flatten_parameters(self):
		self.encoder.flatten_parameters()