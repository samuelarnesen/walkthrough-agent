import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
		self.time_attention = nn.Linear(max_number_of_entries, self.max_number_of_entries)

	def forward(self, encoder_outputs, state, previous_attention):

		sequence_length, batch_size, encoded_size = encoder_outputs.size()

		if type(previous_attention[0]) == type(None):
			previous_attention = torch.zeros([batch_size, sequence_length])
			previous_attention[0, :] = torch.ones([sequence_length])

		weights = self.get_non_normalized_weights(encoder_outputs, state)

		adjusted_previous_weights = self.time_attention(previous_attention)
		truncated_apw = adjusted_previous_weights[:, 0:sequence_length]
		previous_attention_probs = F.softmax(truncated_apw, dim=0)

		#print(encoder_outputs.size(), weights.size(), previous_attention_probs.size(), previous_attention.size())

		adjusted_weights = previous_attention_probs * weights
		normalized_weights_tensor = F.softmax(weights, dim=1)



		attention_applied = torch.bmm(normalized_weights_tensor.unsqueeze(dim=1), encoder_outputs.permute(1, 0, 2))

		return attention_applied.squeeze(dim=1), normalized_weights_tensor