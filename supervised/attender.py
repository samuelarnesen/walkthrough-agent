import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys


class Attender(nn.Module):

	def __init__(self, hidden_size):
		super(Attender, self).__init__()
		self.hidden_size = hidden_size
		self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)

	def attend(self, encoder_output, batch_size, state):
		altered_state = self.attention(state)
		if len(altered_state.size()) == 1:
			altered_state = altered_state.unsqueeze(0).repeat(batch_size, 1)

		weight = torch.bmm(altered_state.unsqueeze(1), encoder_output.unsqueeze(2))
		return weight.squeeze(2)

	def get_non_normalized_weights(self, encoder_outputs, state):

		sequence_length, batch_size, encoded_size = encoder_outputs.size()
		weights = self.attend(encoder_outputs[0, :, :], batch_size, state)
		for i in range(1, sequence_length):
			attention_for_this_word = self.attend(encoder_outputs[i, :, :], batch_size, state)
			weights = torch.cat([weights, attention_for_this_word], dim=1)

		return weights


class BasicAttender(Attender):

	def __init__(self, hidden_size):

		super(BasicAttender, self).__init__(hidden_size)
		self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)

	def forward(self, encoder_outputs, state):

		weights = self.get_non_normalized_weights(encoder_outputs, state)
		normalized_weights_tensor = F.softmax(weights, dim=1)
		attention_applied = torch.bmm(normalized_weights_tensor.unsqueeze(dim=1), encoder_outputs.permute(1, 0, 2))

		return attention_applied.squeeze(dim=1)

class TimeAttender(Attender):

	def __init__(self, hidden_size, max_number_of_entries):

		super(TimeAttender, self).__init__(hidden_size)
		self.max_number_of_entries = max_number_of_entries
		self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)

		kernel_size = int(max_number_of_entries / 14) if int(max_number_of_entries / 14) % 2 == 1 else int(max_number_of_entries / 14) + 1
		self.time_attention = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=int(kernel_size / 2))
		self.time_attention.weight = nn.Parameter(torch.ones([1, 1, self.time_attention.kernel_size[0]], requires_grad=True))
		self.time_attention.bias = nn.Parameter(torch.zeros([1], requires_grad=True))

	def forward(self, encoder_outputs, state, previous_attention, sentence_lengths=None):

		def generate_masks(previous_attention, sentence_lengths):
			bool_tensor = torch.zeros(previous_attention.size(), requires_grad=False)
			for i, length in enumerate(sentence_lengths):
				for j in range(length):
					bool_tensor[i, j] = 1

			return bool_tensor

		sequence_length, batch_size, encoded_size = encoder_outputs.size()

		if type(previous_attention[0]) == type(None):
			previous_attention = torch.zeros([batch_size, sequence_length])
			previous_attention[:, 0] = torch.ones([batch_size])
			previous_attention.requires_grad_()

		weights = self.get_non_normalized_weights(encoder_outputs, state)
		adjusted_previous_weights = F.relu(self.time_attention(previous_attention.unsqueeze(1)).squeeze(1))
		if sentence_lengths != None:
			mask = generate_masks(previous_attention, sentence_lengths)
			adjusted_previous_weights *= mask
		previous_attention_probs = adjusted_previous_weights / torch.sum(adjusted_previous_weights, dim=1).unsqueeze(1)

		adjusted_weights = previous_attention_probs * F.softmax(weights, dim=1)
		normalized_weights_tensor = adjusted_weights / torch.sum(adjusted_weights, dim=1).unsqueeze(dim=1)

		attention_applied = torch.bmm(normalized_weights_tensor.unsqueeze(dim=1), encoder_outputs.permute(1, 0, 2))

		return attention_applied.squeeze(dim=1), normalized_weights_tensor






